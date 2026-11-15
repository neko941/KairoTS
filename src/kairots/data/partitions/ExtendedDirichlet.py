"""Extended Dirichlet, Li and Lyu, NeurIPS2023, arXiv:2311.03154v3.

Omitted coverage, balancing, rounding and minimum-size settings follow
liyipeng00/convergence at5972d5c0692c0f8600711e37630b73ba09efe306,
sim/data/partition.py:ExDirPartitioner. The minimum class coverage uses divisor5.
Both rejection loops retain the author's stopping rules. Data rows are sorted by
sample and step for the store; randomness uses an isolated default_rng(seed).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from kairots.data.partitions._labeled import Concentrated, Coverage, Labeled


class ExtendedDirichlet(Labeled):
    @dataclass(kw_only=True)
    class Config(Coverage.Config, Concentrated.Config):
        name: str = "ExtendedDirichlet"
        alpha: float = 10.0

        def __post_init__(self) -> None:
            if self.num_clients < 1 or self.classes_per_client < 1:
                raise ValueError(
                    "ExtendedDirichlet needs positive client and class counts"
                )
            if not np.isfinite(self.alpha) or self.alpha <= 0:
                raise ValueError("ExtendedDirichlet alpha must be finite and positive")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def clients(self, *, source: Any) -> list[tuple[int, str, pl.DataFrame]]:
        frame, labeled = self.samples(source=source, partition="ExtendedDirichlet")
        labels = sorted(labeled["label"].unique().to_list())
        count = self.config.num_clients
        per_client = self.config.classes_per_client
        if per_client > len(labels) or count * per_client < len(labels):
            raise ValueError(
                "ExtendedDirichlet classes_per_client cannot cover all labels"
            )
        if len(labeled) < 10 * count:
            raise ValueError("ExtendedDirichlet needs at least 10 samples per client")
        generator = np.random.default_rng(self.config.seed)
        minimum = max(per_client * count // len(labels) // 5, 1)
        # shortcut: author rejection loops; revisit if preparation stalls.
        while True:
            members: dict[Any, list[int]] = {label: [] for label in labels}
            for client_id in range(count):
                for index in generator.choice(len(labels), per_client, replace=False):
                    members[labels[index]].append(client_id)
            if min(map(len, members.values())) >= minimum:
                break
        available = [0] * count
        for label, member_ids in members.items():
            size = labeled.filter(pl.col("label") == label).height
            for client_id in member_ids:
                available[client_id] += size
        if min(available) < 10:
            raise ValueError(
                "ExtendedDirichlet assigned labels cannot supply 10 samples per client"
            )
        while True:
            assigned: list[list[Any]] = [[] for _ in range(count)]
            for label in labels:
                sample_ids = labeled.filter(pl.col("label") == label)[
                    "sample"
                ].to_list()
                generator.shuffle(sample_ids)
                proportions = generator.dirichlet([self.config.alpha] * count)
                proportions *= [
                    len(assigned[client_id]) < len(labeled) / count
                    and client_id in members[label]
                    for client_id in range(count)
                ]
                total = proportions.sum()
                if not np.isfinite(total) or total <= 0:
                    raise ValueError("ExtendedDirichlet balanced proportions vanished")
                cuts = (np.cumsum(proportions / total) * len(sample_ids)).astype(int)[
                    :-1
                ]
                if cuts.size and cuts[-1] != len(sample_ids):
                    cuts[members[label][-1] :] = len(sample_ids)
                for client_id, sample_slice in enumerate(np.split(sample_ids, cuts)):
                    assigned[client_id].extend(sample_slice.tolist())
            if min(map(len, assigned)) >= 10:
                return self.materialize(source=source, frame=frame, assigned=assigned)
