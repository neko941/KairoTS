from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from kairots.config.schema import Experiment


def write_ett_sources(*, path: Path, short_etth1: bool = False) -> None:
    for variant, offset in (("ETTh1", 0.0), ("ETTh2", 2.0)):
        directory = path / variant
        directory.mkdir(parents=True)
        count = 14 if short_etth1 and variant == "ETTh1" else 240
        time = np.arange(count)
        pl.DataFrame(
            {
                "date": time.tolist(),
                **{
                    channel: (np.sin(time / 9 + offset + index) + offset).tolist()
                    for index, channel in enumerate(
                        ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
                    )
                },
            }
        ).write_csv(directory / f"{variant}.csv")


def ett_experiment(
    *,
    raw_path: Path,
    runs_dir: Path,
    strategy: str = "FedAvg",
    iterations: int = 1,
    executor: str = "Serial",
    min_windows: int = 2,
    save_checkpoints: bool = False,
    optimizer: str = "Adam",
) -> Experiment:
    return Experiment.from_dict(
        values={
            "data": {
                "split": {
                    "input_length": 8,
                    "output_length": 3,
                    "ratios": [0.6, 0.2, 0.2],
                    "min_windows": min_windows,
                },
                "source": {
                    "name": "ETDatasetHour",
                    "raw_path": str(raw_path),
                    "download": False,
                    "batch_size": 16,
                },
                "store": {"root": str(runs_dir.parent / "datasets")},
            },
            "strategy": {"name": strategy, "iterations": iterations},
            "learner": {"learning_rate": 0.001, "optimizer": {"name": optimizer}},
            "runtime": {
                "name": executor,
                "device": "cpu",
                "num_workers": 2,
                "seed": 53,
                "runs_dir": str(runs_dir),
                "save_checkpoints": save_checkpoints,
            },
        }
    )
