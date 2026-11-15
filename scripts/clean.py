"""Delete regenerable caches: __pycache__, pytest/ruff caches, build output.

Never touches runs/, datasets/ or .conda/; `--prepared` also deletes every
`datasets/<Family>/prepared/` directory, which the store rebuilds from raw files
on the next run. `src/*.egg-info` is a setuptools build
leftover: the editable install lives in site-packages (.pth + dist-info).
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
DIRECTORIES = ("__pycache__", ".pytest_cache", ".pytest-tmp", ".ruff_cache", "build")
FILES = ("*.pyc", "*.pyo", ".coverage")
KEEP = {".conda", ".git", "runs", "datasets", "tmp"}


def targets(*, prepared: bool) -> list[Path]:
    found: list[Path] = []
    pending = [ROOT]
    while pending:
        directory = pending.pop()
        try:
            children = list(directory.iterdir())
        except PermissionError:
            continue
        for child in children:
            if child.name in KEEP:
                continue
            if child.is_dir() and not child.is_symlink():
                if child.name in DIRECTORIES or child.name.endswith(".egg-info"):
                    found.append(child)
                else:
                    pending.append(child)
            elif any(child.match(pattern) for pattern in FILES):
                found.append(child)
    if prepared:
        found.extend((ROOT / "datasets").glob("*/prepared"))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be deleted"
    )
    parser.add_argument(
        "--prepared", action="store_true", help="also delete prepared datasets"
    )
    args = parser.parse_args()
    dry_run = args.dry_run
    failed = 0
    for path in targets(prepared=args.prepared):
        print(("would delete " if dry_run else "delete ") + str(path.relative_to(ROOT)))
        if dry_run:
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as error:
            failed += 1
            print(f"  failed: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
