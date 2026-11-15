"""Mean squared error over scored positions, as a batch loss or mergeable metric.

A non-finite target marks a position that is not scored (imputation: not hidden, or
natively missing). With every target finite this equals `mse_loss`.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.metrics._mean import add, mean, merge


class MSE:
    ROLES = ("loss", "metric")
    TASKS = (
        "Forecasting",
        "Imputation",
        "MaskedImputation",
        "AnomalyDetection",
        "Generation",
    )
    NEEDS = ("point_prediction", "target")
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config:
        name: str = "MSE"

    def __init__(self, *, config: Config | None = None) -> None:
        self.total: Tensor | None = None
        self.count: Tensor | None = None

    def __call__(self, *, prediction: Tensor, target: Tensor) -> Tensor:
        scored = target.isfinite()
        error = torch.where(scored, prediction - target, 0.0)
        return error.square().sum() / scored.sum()

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        scored = target.isfinite()
        error = torch.where(scored, prediction.double() - target.double(), 0.0)
        add(metric=self, total=error.square().sum(), count=scored.sum())

    def merge(self, *, other: "MSE") -> None:
        merge(metric=self, other=other)

    def compute(self) -> float:
        return mean(metric=self)
