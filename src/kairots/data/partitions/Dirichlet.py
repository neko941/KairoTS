"""Classwise label skew, Yurochkin et al., ICML2019, arXiv:1905.12022v1.

The balancing step follows IBM/probabilistic-federated-neural-matching at
f44cf4281944fae46cdce1b8bc7cde3e7c44bd70, experiment.py:partition_data.
Each label draws client proportions; clients already at the average size are
excluded from that label's draw. Hsu et al.'s clientwise class-mixture draw is
a different partition. The minimum client size is configurable (official:10),
with at most100 rejection draws. Randomness uses an isolated default_rng.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from kairots.data.partitions._labeled import Concentrated, Labeled


@dataclass(kw_only=True)
class Dirichlet(Labeled):
    @dataclass(kw_only=True)
    class Config(Concentrated.Config):
        name: str = "Dirichlet"
        min_samples: int = 2

        def __post_init__(self) -> None:
            if self.num_clients < 1 or self.min_samples < 1:
                raise ValueError("Dirichlet needs positive num_clients and min_samples")
            if not np.isfinite(self.alpha) or self.alpha <= 0:
                raise ValueError("Dirichlet alpha must be finite and positive")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def clients(self, *, source: Any) -> list[tuple[int, str, pl.DataFrame]]:
        frame, labeled = self.samples(source=source, partition="Dirichlet")
        if len(labeled) < self.config.num_clients * self.config.min_samples:
            raise ValueError("Dirichlet has too few samples to meet min_samples")
        labels = sorted(labeled.get_column("label").unique().to_list())
        generator = np.random.default_rng(self.config.seed)
        for _ in range(100):
            assigned: list[list[Any]] = [[] for _ in range(self.config.num_clients)]
            for label in labels:
                sample_ids = (
                    labeled.filter(pl.col("label") == label)
                    .get_column("sample")
                    .to_list()
                )
                generator.shuffle(sample_ids)
                proportions = generator.dirichlet(
                    [self.config.alpha] * self.config.num_clients
                )
                proportions *= [
                    len(client_samples) < len(labeled) / self.config.num_clients
                    for client_samples in assigned
                ]
                total = proportions.sum()
                if not np.isfinite(total) or total <= 0:
                    raise ValueError("Dirichlet proportions underflow; increase alpha")
                proportions /= total
                cuts = (np.cumsum(proportions) * len(sample_ids)).astype(int)[:-1]
                for client_id, sample_slice in enumerate(np.split(sample_ids, cuts)):
                    assigned[client_id].extend(sample_slice.tolist())
            if all(
                len(sample_ids) >= self.config.min_samples for sample_ids in assigned
            ):
                return self.materialize(source=source, frame=frame, assigned=assigned)
        raise ValueError(
            f"Could not meet min_samples={self.config.min_samples} "
            f"with alpha={self.config.alpha} after 100 draws"
        )
