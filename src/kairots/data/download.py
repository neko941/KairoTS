"""Download raw dataset files declared by sources.

A source lists its files as `RemoteFile`s; `fetch` is the only download code. Each
file is streamed to `.<name>.part`, resumed with an HTTP Range request after an
interruption, tried against every mirror with retries, checked against its sha256,
and only then renamed into place. A file that is already in place is not fetched.
Archives are unpacked into a directory next to them, with members that would escape it
rejected.
"""

import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

ATTEMPTS = 3
TIMEOUT_SECONDS = 60
BLOCK_BYTES = 1 << 20
USER_AGENT = "kairots-downloader"


@dataclass(frozen=True, kw_only=True)
class RemoteFile:
    """One raw file: where it goes under the source's raw directory, and its origin.

    Args:
        path: Location relative to the source's `raw_path`.
        urls: Mirrors, tried in order.
        sha256: Expected digest; `None` skips the check (avoid for published data).
        extract: For an archive, the directory (relative to `raw_path`) to unpack into.
    """

    path: str
    urls: tuple[str, ...]
    sha256: str | None = None
    extract: str | None = None


class ChecksumError(ValueError):
    """Downloaded content does not match the expected digest."""


def fetch(*, remote: RemoteFile, root: Path) -> Path:
    """Make `root / remote.path` present and verified; return the usable path."""
    target = root / remote.path
    if remote.extract is not None and (root / remote.extract).is_dir():
        return root / remote.extract
    if not target.is_file():
        errors: list[str] = []
        failure_type: type[Exception] = ConnectionError
        for url in remote.urls:
            for attempt in range(1, ATTEMPTS + 1):
                try:
                    _stream(url=url, target=target, sha256=remote.sha256)
                    break
                except ChecksumError as error:
                    failure_type = ChecksumError
                    errors.append(f"{url}: {error}")
                    break
                except (urllib.error.URLError, OSError, TimeoutError) as error:
                    failure_type = ConnectionError
                    errors.append(f"{url} (attempt {attempt}): {error}")
                    permanent = url.startswith("file:") or (
                        isinstance(error, urllib.error.HTTPError) and error.code < 500
                    )
                    if permanent:
                        break  # a 4xx or a missing local file will not change
                    time.sleep(2**attempt if attempt < ATTEMPTS else 0)
            if target.is_file():
                break
        else:
            raise failure_type(
                f"could not download {remote.path}:\n  " + "\n  ".join(errors)
            )
    if remote.extract is None:
        return target
    unpacked = _extract(archive=target, directory=root / remote.extract)
    target.unlink()
    return unpacked


def _stream(*, url: str, target: Path, sha256: str | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # A fixed .part name lets a rerun resume, but two processes downloading
    # the same file at once would clash; add a lock file if that ever happens.
    partial = target.with_name(f".{target.name}.part")
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        # Servers (and file://) that ignore Range send the whole file: start over.
        resumed = offset > 0 and getattr(response, "status", None) == 206
        total = response.headers.get("Content-Length")
        expected = int(total) + (offset if resumed else 0) if total else None
        with partial.open("ab" if resumed else "wb") as output:
            written = offset if resumed else 0
            reported = 0
            for block in iter(lambda: response.read(BLOCK_BYTES), b""):
                output.write(block)
                written += len(block)
                if expected and written * 10 // expected > reported:
                    reported = written * 10 // expected
                    print(
                        f"  {target.name}: {written / 1e6:.1f}/{expected / 1e6:.1f} MB",
                        file=sys.stderr,
                    )
    if expected is not None and partial.stat().st_size != expected:
        raise OSError(f"{url} ended early at {partial.stat().st_size} of {expected}")
    if sha256 is not None:
        digest = _sha256(path=partial)
        if digest != sha256:
            partial.unlink()
            raise ChecksumError(f"{url} sha256 {digest} does not match {sha256}")
    os.replace(partial, target)


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(*, archive: Path, directory: Path) -> Path:
    build = directory.with_name(f".{directory.name}-{uuid.uuid4().hex[:8]}")
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if not (build / member).resolve().is_relative_to(build.resolve()):
                    raise ValueError(f"{archive.name} member escapes target: {member}")
            bundle.extractall(build)
        os.replace(build, directory)
    finally:
        shutil.rmtree(build, ignore_errors=True)
    return directory
