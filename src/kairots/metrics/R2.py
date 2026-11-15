"""Mergeable coefficient of determination, averaged over channels. Non-finite
targets (positions imputation does not score) are left out per channel.
Channels without observations are omitted. Any scored constant-target channel
makes the aggregate undefined (NaN), including a perfectly predicted constant.

Test-only: R2 is for reporting, so it cannot be selected for the train or validation
phase and never drives model selection.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class R2:
    ROLES = ("metric",)
    TASKS = ("Forecasting", "Imputation", "MaskedImputation")
    NEEDS = ("point_prediction", "target")
    DIRECTION = "max"
    PHASES = ("test",)

    squared_error: Tensor | None = None
    target_sum: Tensor | None = None
    target_squared_sum: Tensor | None = None
    count: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "R2"

    def __init__(self, *, config: Config | None = None) -> None:
        self.squared_error = None
        self.target_sum = None
        self.target_squared_sum = None
        self.count = None

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        # Sums per channel (last axis) in float64, so merging clients stays exact.
        target = target.double().flatten(end_dim=-2)
        prediction = prediction.double().flatten(end_dim=-2)
        scored = target.isfinite()
        target = target.where(scored, 0.0)
        error = (prediction - target).where(scored, 0.0)
        sums = (
            (error**2).sum(dim=0),
            target.sum(dim=0),
            (target**2).sum(dim=0),
        )
        self._add(sums=sums, count=scored.sum(dim=0).double())

    def merge(self, *, other: R2) -> None:
        if other.squared_error is None:
            return
        assert other.target_sum is not None and other.target_squared_sum is not None
        assert other.count is not None
        self._add(
            sums=(other.squared_error, other.target_sum, other.target_squared_sum),
            count=other.count,
        )

    def _add(self, *, sums: tuple[Tensor, Tensor, Tensor], count: Tensor) -> None:
        if self.squared_error is None or self.count is None:
            self.squared_error, self.target_sum, self.target_squared_sum = sums
            self.count = count
            return
        assert self.target_sum is not None and self.target_squared_sum is not None
        self.squared_error = self.squared_error + sums[0]
        self.target_sum = self.target_sum + sums[1]
        self.target_squared_sum = self.target_squared_sum + sums[2]
        self.count = self.count + count

    def compute(self) -> float:
        if self.squared_error is None or self.count is None or not self.count.any():
            raise ValueError("R2 has no observations")
        assert self.target_sum is not None and self.target_squared_sum is not None
        scored = self.count > 0
        total = (
            self.target_squared_sum[scored]
            - self.target_sum[scored] ** 2 / self.count[scored]
        )
        if (total <= 0).any():
            return float("nan")
        return float(torch.mean(1 - self.squared_error[scored] / total).item())
