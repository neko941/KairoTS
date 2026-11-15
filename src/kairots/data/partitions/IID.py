"""Deal samples to clients uniformly at random (independent, identically
distributed)."""

import random
from dataclasses import dataclass
from typing import Any

import polars as pl

from kairots.data.partitions._labeled import Labeled


@dataclass(kw_only=True)
class IID:
    @dataclass(kw_only=True)
    class Config(Labeled.Config):
        name: str = "IID"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def clients(self, *, source: Any) -> list[tuple[int, str, pl.DataFrame]]:
        frame = pl.concat([frame for _, frame in source.clients])
        if "sample" not in frame.columns:
            raise ValueError("IID deals samples; use PerFile for continuous series")
        sample_ids = frame.get_column("sample").unique(maintain_order=True).to_list()
        if len(sample_ids) < self.config.num_clients:
            raise ValueError(
                f"{len(sample_ids)} samples cannot fill "
                f"{self.config.num_clients} clients"
            )
        random.Random(self.config.seed).shuffle(sample_ids)
        return [
            (
                client_id,
                f"{source.name}/{client_id}",
                frame.filter(
                    pl.col("sample").is_in(
                        sample_ids[client_id :: self.config.num_clients]
                    )
                ).sort("sample", "step"),
            )
            for client_id in range(self.config.num_clients)
        ]
