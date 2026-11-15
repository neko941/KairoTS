"""Mergeable coverage error of the outer central interval: `|c - nominal|`, where `c`
is the share of scored targets inside `[q_first, q_last]` and `nominal` is
`q_last - q_first` (`(n - 1) / (n + 1)` for `n` evenly spaced levels, 0.8 for 9).
0 means the interval covers as often as it claims; it says nothing about width
(`IntervalWidth`)."""

from dataclasses import dataclass

import torch
from torch import Tensor


def central_interval(*, prediction: Tensor) -> tuple[Tensor, Tensor, float]:
    """(lower, upper, nominal coverage) from a quantile prediction's outer levels."""
    num_quantiles = prediction.shape[-1]
    if num_quantiles < 2:
        raise ValueError("an interval needs num_quantiles >= 2")
    return (
        prediction[..., 0],
        prediction[..., -1],
        (num_quantiles - 1) / (num_quantiles + 1),
    )


@dataclass(kw_only=True)
class CoverageError:
    ROLES = ("metric",)
    TASKS = ("ProbabilisticForecasting",)
    NEEDS = ("quantile_prediction", "target")
    DIRECTION = "min"

    covered: Tensor | None = None
    count: Tensor | None = None
    nominal: float = 0.0

    @dataclass(kw_only=True)
    class Config:
        name: str = "CoverageError"

    def __init__(self, *, config: Config | None = None) -> None:
        self.covered = None
        self.count = None
        self.nominal = 0.0

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        lower, upper, self.nominal = central_interval(prediction=prediction)
        scored = target.isfinite()
        covered = (scored & (target >= lower) & (target <= upper)).sum()
        count = scored.sum()
        self.covered = covered if self.covered is None else self.covered + covered
        self.count = count if self.count is None else self.count + count

    def merge(self, *, other: CoverageError) -> None:
        if other.covered is None or other.count is None:
            return
        self.nominal = other.nominal
        self.covered = (
            other.covered if self.covered is None else self.covered + other.covered
        )
        self.count = other.count if self.count is None else self.count + other.count

    def compute(self) -> float:
        if self.covered is None or self.count is None or not self.count:
            raise ValueError("CoverageError has no observations")
        coverage = self.covered.double() / self.count
        return float(torch.abs(coverage - self.nominal).item())
