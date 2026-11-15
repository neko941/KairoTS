"""Linux benchmark of one ETTh1 window trace through NPZ, mmap NPY and Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

try:
    import resource
except ImportError:  # pragma: no cover - Windows cannot run this Linux benchmark.
    resource = None  # type: ignore[assignment]

WINDOW = 120
BATCH = 32
TRACE = 2048
REPEATS = 3


def _load(*, path: str, kind: str) -> np.ndarray:
    if kind == "npz":
        with np.load(path, allow_pickle=False) as archive:
            return archive["values"]
    if kind == "npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    return pl.read_parquet(path).to_numpy()


def _windows(*, values: np.ndarray, starts: np.ndarray) -> float:
    total = 0.0
    for start in starts:
        total += float(values[start : start + WINDOW].sum(dtype=np.float64))
    return total


def _rss_mib() -> float:
    resource_api = cast(Any, resource)
    if resource_api is None:
        raise RuntimeError("peak RSS collection requires Linux resource.getrusage")
    return resource_api.getrusage(resource_api.RUSAGE_SELF).ru_maxrss / 1024


def _request_cache_eviction(*, path: Path) -> str:
    with path.open("rb") as handle:
        os_api = cast(Any, os)
        advise = getattr(os_api, "posix_fadvise", None)
        dontneed = getattr(os_api, "POSIX_FADV_DONTNEED", None)
        if callable(advise) and dontneed is not None:
            advise(handle.fileno(), 0, 0, dontneed)
            return "POSIX_FADV_DONTNEED requested; eviction remains advisory"
    return "unavailable"


def _worker(
    *, kind: str, path: Path, ordered: np.ndarray, random: np.ndarray
) -> dict[str, Any]:
    cache_hint = _request_cache_eviction(path=path)
    started = time.perf_counter()
    values = _load(path=str(path), kind=kind)
    first = _windows(values=values, starts=ordered[:BATCH])
    first_batch_seconds = time.perf_counter() - started
    advised_cold_sweeps = []
    for trace in (ordered, random):
        cache_hint = _request_cache_eviction(path=path)
        started = time.perf_counter()
        cold_values = _load(path=str(path), kind=kind)
        checksum = _windows(values=cold_values, starts=trace)
        elapsed = time.perf_counter() - started
        advised_cold_sweeps.append(
            {
                "windows_per_second": len(trace) / max(1e-12, elapsed),
                "checksum": checksum,
            }
        )
    started = time.perf_counter()
    for _ in range(REPEATS):
        for trace in (ordered, random):
            _windows(values=values, starts=trace)
    warm_seconds = time.perf_counter() - started
    return {
        "kind": kind,
        "first_batch_seconds": first_batch_seconds,
        "advised_cold_sweeps": advised_cold_sweeps,
        "cache_hint": cache_hint,
        "warm_windows_per_second": (len(ordered) + len(random))
        * REPEATS
        / max(1e-12, warm_seconds),
        "peak_rss_mib": _rss_mib(),
        "checksums": [first, *(sweep["checksum"] for sweep in advised_cold_sweeps)],
        "array_sha256": hashlib.sha256(np.asarray(values).tobytes()).hexdigest(),
    }


def _ray_worker(path: str, kind: str, starts: list[int]) -> dict[str, float]:
    values = _load(path=path, kind=kind)
    started = time.perf_counter()
    total = _windows(values=values, starts=np.asarray(starts, dtype=np.int64))
    elapsed = time.perf_counter() - started
    return {"windows_per_second": len(starts) / max(1e-12, elapsed), "checksum": total}


def _benchmark(*, source: Path, output: Path) -> dict[str, Any]:
    source_read_started = time.perf_counter()
    frame = pl.read_csv(source, try_parse_dates=True)
    values = frame.select(frame.columns[1:]).cast(pl.Float32).to_numpy()
    values = np.ascontiguousarray(values, dtype=np.float32)
    source_read_seconds = time.perf_counter() - source_read_started
    if values.shape[0] <= WINDOW or not np.isfinite(values).all():
        raise ValueError("ETTh1 input must have finite rows beyond the window length")

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "reference.npy", values, allow_pickle=False)
    starts = np.arange(values.shape[0] - WINDOW + 1, dtype=np.int64)
    random = np.random.default_rng(941).choice(starts, size=TRACE, replace=True)
    ordered = starts[:TRACE]
    np.savez_compressed(output / "trace.npz", ordered=ordered, random=random)

    paths = {
        "npz": output / "values.npz",
        "npy": output / "values.npy",
        "parquet": output / "values.parquet",
    }
    prep_seconds: dict[str, float] = {}
    started = time.perf_counter()
    np.savez_compressed(paths["npz"], values=values)
    prep_seconds["npz"] = time.perf_counter() - started
    started = time.perf_counter()
    mapped = np.lib.format.open_memmap(
        paths["npy"], mode="w+", dtype=np.float32, shape=values.shape
    )
    mapped[:] = values
    mapped.flush()
    del mapped
    prep_seconds["npy"] = time.perf_counter() - started
    started = time.perf_counter()
    pl.DataFrame(values, schema=frame.columns[1:], orient="row").write_parquet(
        paths["parquet"]
    )
    prep_seconds["parquet"] = time.perf_counter() - started

    results: dict[str, Any] = {}
    for kind, path in paths.items():
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            kind,
            str(path),
            str(output / "trace.npz"),
        ]
        result = json.loads(subprocess.check_output(command, text=True))
        result["prep_seconds"] = prep_seconds[kind]
        result["bytes"] = path.stat().st_size
        results[kind] = result

    source_hash = hashlib.sha256(values.tobytes()).hexdigest()
    array_hashes = [results[kind]["array_sha256"] for kind in paths]
    if array_hashes != [source_hash] * len(paths):
        raise AssertionError("storage formats changed array values or dtype")
    checksums = [results[kind]["checksums"] for kind in paths]
    if not np.allclose(checksums, checksums[0], rtol=0, atol=1e-4):
        raise AssertionError("storage formats returned different windows")

    import ray

    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        logging_level=logging.ERROR,
    )
    try:
        remote = ray.remote(_ray_worker)
        ray_results = {}
        for kind, path in paths.items():
            tasks = [
                remote.remote(str(path), kind, ordered[:TRACE].tolist()),
                remote.remote(str(path), kind, random[:TRACE].tolist()),
            ]
            ray_results[kind] = ray.get(tasks)
        results["ray_two_workers"] = ray_results
    finally:
        ray.shutdown()

    ray_checksums = [
        ray_results[kind][index]["checksum"] for kind in paths for index in range(2)
    ]
    expected = [ray_results["npz"][index]["checksum"] for index in range(2)]
    if not np.allclose(ray_checksums, expected * len(paths), rtol=0, atol=1e-4):
        raise AssertionError("Ray workers returned different window traces")

    results["trace"] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "window": WINDOW,
        "batch": BATCH,
        "ordered_windows": int(len(ordered)),
        "random_windows": int(len(random)),
        "repeats": REPEATS,
        "seed": 941,
        "source_read_seconds": source_read_seconds,
        "source_sha256": source_hash,
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--worker", nargs=3, metavar=("KIND", "PATH", "TRACE"))
    args = parser.parse_args()
    if args.worker:
        kind, path, trace_path = args.worker
        with np.load(trace_path, allow_pickle=False) as trace:
            report = _worker(
                kind=kind,
                path=Path(path),
                ordered=trace["ordered"],
                random=trace["random"],
            )
        print(json.dumps(report))
        return 0
    if args.source is None or args.output is None:
        parser.error("source and output are required")
    report = _benchmark(source=args.source, output=args.output)
    report_path = args.output / "storage-benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
