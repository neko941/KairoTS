"""PFLlib class-count label skew, Zhang et al., JMLR2025, arXiv:2312.04992v2.

The original pat branch is pinned at0169ba7e412c9856a08bb3faefab1e35f538a3c1,
dataset/utils/dataset_utils.py:separate_data. It assigns classes to the first
eligible clients and splits each class equally or with random sizes. This is
distinct from McMahan's equal-shard partition. Original batch10/train_ratio0.75
constants define the random-size lower bound only; the split remains independent.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from kairots.data.partitions._labeled import Coverage, Labeled


class ClassPerClient(Labeled):
    @dataclass(kw_only=True)
    class Config(Coverage.Config):
        name: str = "ClassPerClient"
        balance: bool = False

        def __post_init__(self) -> None:
            if self.num_clients < 1 or self.classes_per_client < 1:
                raise ValueError(
                    "ClassPerClient needs positive client and class counts"
                )

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def clients(self, *, source: Any) -> list[tuple[int, str, pl.DataFrame]]:
        frame, labeled = self.samples(source=source, partition="ClassPerClient")
        labels = sorted(labeled["label"].unique().to_list())
        count = self.config.num_clients
        per_client = self.config.classes_per_client
        if not labels or per_client > len(labels) or count * per_client < len(labels):
            raise ValueError(
                "ClassPerClient classes_per_client cannot cover all labels"
            )
        remaining = [per_client] * count
        assigned: list[list[Any]] = [[] for _ in range(count)]
        allocation_limit = math.ceil(count / len(labels) * per_client)
        least_samples = int(min(40.0, len(labeled) / count / 2))
        generator = np.random.default_rng(self.config.seed)
        for label in labels:
            selected = [
                client_id for client_id in range(count) if remaining[client_id]
            ][:allocation_limit]
            if not selected:
                raise ValueError(
                    "ClassPerClient exhausted slots before covering all labels"
                )
            sample_ids = labeled.filter(pl.col("label") == label)["sample"].to_list()
            average = len(sample_ids) / len(selected)
            if self.config.balance:
                sizes = [int(average)] * (len(selected) - 1)
            elif len(selected) == 1:
                sizes = []
            else:
                lower = int(max(average / 10, least_samples / len(labels)))
                upper = int(average)
                if lower >= upper:
                    raise ValueError("ClassPerClient has no valid random-size interval")
                sizes = generator.integers(lower, upper, len(selected) - 1).tolist()
            sizes.append(len(sample_ids) - sum(sizes))
            start = 0
            for client_id, size in zip(selected, sizes, strict=True):
                assigned[client_id].extend(sample_ids[start : start + size])
                remaining[client_id] -= 1
                start += size
        if not all(assigned):
            raise ValueError(
                "ClassPerClient produced an empty client; use more samples"
            )
        return self.materialize(source=source, frame=frame, assigned=assigned)
