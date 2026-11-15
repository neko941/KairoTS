"""Mean absolute error as a batch loss or mergeable sample-pooled metric.

A non-finite target marks an unscored position (see `metrics/MSE.py`).
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class MAE:
    ROLES = ("loss", "metric")
    TASKS = ("Forecasting", "Imputation", "MaskedImputation")
    NEEDS = ("point_prediction", "target")
    DIRECTION = "min"

    absolute_error: Tensor | None = None
    count: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "MAE"

    def __init__(self, *, config: Config | None = None) -> None:
        self.absolute_error = None
        self.count = None

    @staticmethod
    def _sum(*, prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        scored = target.isfinite()
        return torch.where(scored, prediction - target, 0.0).abs().sum(), scored.sum()

    def __call__(self, *, prediction: Tensor, target: Tensor) -> Tensor:
        total, count = self._sum(prediction=prediction, target=target)
        if not count:
            raise ValueError("MAE has no scored targets")
        return total / count

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        value, count = self._sum(prediction=prediction, target=target)
        self.absolute_error = (
            value if self.absolute_error is None else self.absolute_error + value
        )
        self.count = count if self.count is None else self.count + count

    def merge(self, *, other: MAE) -> None:
        if other.absolute_error is not None:
            self.absolute_error = (
                other.absolute_error
                if self.absolute_error is None
                else self.absolute_error + other.absolute_error
            )
        if other.count is not None:
            self.count = other.count if self.count is None else self.count + other.count

    def compute(self) -> float:
        if self.absolute_error is None or self.count is None or not self.count:
            raise ValueError("MAE has no observations")
        return float((self.absolute_error / self.count).item())
