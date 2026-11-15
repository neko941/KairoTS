"""Mergeable symmetric mean absolute percentage error, TSLib `utils/m4_summary.py`
`smape_2`: `200 * |prediction - target| / (|target| + |prediction|)`, a zero
denominator counts as 1, pooled over scored positions (TSLib takes the mean over every
element of the group). A non-finite target marks an unscored position.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class SMAPE:
    ROLES = ("metric",)
    TASKS = ("Forecasting",)
    NEEDS = ("point_prediction", "target")
    DIRECTION = "min"

    total: Tensor | None = None
    count: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "SMAPE"

    def __init__(self, *, config: Config | None = None) -> None:
        self.total = None
        self.count = None

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        scored = target.isfinite()
        target = torch.where(scored, target, 0.0)
        prediction = torch.where(scored, prediction, 0.0)
        denominator = target.abs() + prediction.abs()
        denominator = torch.where(denominator == 0, 1.0, denominator)
        value = (200 * (prediction - target).abs() / denominator).sum()
        count = scored.sum()
        self.total = value if self.total is None else self.total + value
        self.count = count if self.count is None else self.count + count

    def merge(self, *, other: SMAPE) -> None:
        if other.total is not None:
            self.total = other.total if self.total is None else self.total + other.total
        if other.count is not None:
            self.count = other.count if self.count is None else self.count + other.count

    def compute(self) -> float:
        if self.total is None or self.count is None or not self.count:
            raise ValueError("SMAPE has no observations")
        return float((self.total / self.count).item())
