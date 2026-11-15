"""Distance to closest record, a memorization check: the mean Euclidean distance from
each generated window to its nearest train window of the same client, over the same
mean for each real held-out window. A ratio well below 1 means the samples sit closer
to the training data than unseen real data does. Merging adds the sums.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class DCR:
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target", "train")
    PHASES = ("test",)
    DIRECTION = "max"

    generated_total: float = 0.0
    generated_count: int = 0
    real_total: float = 0.0
    real_count: int = 0

    @dataclass(kw_only=True)
    class Config:
        name: str = "DCR"

    def __init__(self, *, config: Config | None = None) -> None:
        self.generated_total = 0.0
        self.generated_count = 0
        self.real_total = 0.0
        self.real_count = 0

    def update(self, *, prediction: Tensor, target: Tensor, train: Tensor) -> None:
        records = train.flatten(1).double()

        def closest(*, windows: Tensor) -> Tensor:
            return torch.cdist(windows.flatten(1).double(), records).min(dim=1).values

        self.generated_total += float(closest(windows=prediction).sum())
        self.generated_count += prediction.shape[0]
        self.real_total += float(closest(windows=target).sum())
        self.real_count += target.shape[0]

    def merge(self, *, other: DCR) -> None:
        self.generated_total += other.generated_total
        self.generated_count += other.generated_count
        self.real_total += other.real_total
        self.real_count += other.real_count

    def compute(self) -> float:
        if not self.generated_count or not self.real_count or not self.real_total:
            raise ValueError("DCR has no observations")
        return (self.generated_total / self.generated_count) / (
            self.real_total / self.real_count
        )
