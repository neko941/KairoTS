"""Read labeled panels and materialize disjoint sample assignments."""

from dataclasses import dataclass
from typing import Any

import polars as pl


class Labeled:
    """Shared labeled partition operations included in prepared-data fingerprints."""

    @dataclass(kw_only=True)
    class Config:
        name: str = "Labeled"
        num_clients: int = 2
        seed: int = 0

    @staticmethod
    def samples(*, source: Any, partition: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        if not source.clients:
            raise ValueError(f"{partition} has no samples")
        frame = pl.concat([frame for _, frame in source.clients])
        if "sample" not in frame.columns:
            raise ValueError(
                f"{partition} deals samples; use PerFile for continuous series"
            )
        if "label" not in frame.columns:
            raise ValueError(
                f"{partition} deals labeled samples; source needs a label column"
            )
        if frame["sample"].null_count():
            raise ValueError(f"{partition} requires a sample ID for every row")
        if frame["label"].null_count():
            raise ValueError(f"{partition} requires a label for every sample")
        unique = frame.select("sample", "label").unique(maintain_order=True)
        if unique["sample"].n_unique() != len(unique):
            raise ValueError(f"{partition} requires exactly one label per sample")
        return frame, unique

    @staticmethod
    def materialize(
        *, source: Any, frame: pl.DataFrame, assigned: list[list[Any]]
    ) -> list[tuple[int, str, pl.DataFrame]]:
        return [
            (
                client_id,
                f"{source.name}/{client_id}",
                frame.filter(pl.col("sample").is_in(sample_ids)).sort("sample", "step"),
            )
            for client_id, sample_ids in enumerate(assigned)
        ]


class Coverage:
    @dataclass(kw_only=True)
    class Config(Labeled.Config):
        name: str = "Coverage"
        classes_per_client: int = 2


class Concentrated:
    @dataclass(kw_only=True)
    class Config(Labeled.Config):
        name: str = "Concentrated"
        alpha: float = 0.5
