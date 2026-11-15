"""Label-sorted shards, McMahan et al., AISTATS2017, arXiv:1602.05629v4.

Section3 divides sorted MNIST labels into200 equal shards and randomly gives
two to each of100 clients. Shard counts generalize through num_clients and
shards_per_client. Within-label ties preserve source order; randomness uses an
isolated default_rng(seed). Indivisible sample counts fail rather than lose rows.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from kairots.data.partitions._labeled import Labeled


class Pathological(Labeled):
    @dataclass(kw_only=True)
    class Config(Labeled.Config):
        name: str = "Pathological"
        shards_per_client: int = 2

        def __post_init__(self) -> None:
            if self.num_clients < 1 or self.shards_per_client < 1:
                raise ValueError("Pathological needs positive client and shard counts")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def clients(self, *, source: Any) -> list[tuple[int, str, pl.DataFrame]]:
        frame, labeled = self.samples(source=source, partition="Pathological")
        count = self.config.num_clients * self.config.shards_per_client
        if not len(labeled) or len(labeled) % count:
            raise ValueError(
                "Pathological needs equal nonempty shards; "
                "sample count must be divisible by total shards"
            )
        sample_ids = labeled.sort("label", maintain_order=True)["sample"].to_list()
        size = len(sample_ids) // count
        order = np.random.default_rng(self.config.seed).permutation(count)
        assigned = [
            [
                sample
                for shard in shards
                for sample in sample_ids[shard * size : (shard + 1) * size]
            ]
            for shards in order.reshape(
                self.config.num_clients, self.config.shards_per_client
            )
        ]
        return self.materialize(source=source, frame=frame, assigned=assigned)
