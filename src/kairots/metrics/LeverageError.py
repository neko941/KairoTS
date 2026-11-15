"""Mergeable leverage-effect check: the mean absolute difference between the real and
generated `corr(x[t], x[t + tau]^2)` at lags `tau` 1 to `LAGS` per channel, each
pooled over the step pairs within windows (a stylized fact of returns: falls raise
later volatility, so the real correlation is negative at short lags). Merging adds
the sufficient statistics.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def lagged_statistics(*, windows: Tensor, lags: int) -> Tensor:
    """(6, lags, channels): count and the sums of a, b, a^2, b^2 and ab for the pairs
    `a = x[t]`, `b = x[t + tau]^2`."""
    values = windows.double()
    rows = []
    for lag in range(1, lags + 1):
        before = values[:, :-lag].flatten(0, 1)
        after = values[:, lag:].flatten(0, 1).square()
        rows.append(
            torch.stack(
                [
                    torch.full_like(before[0], before.shape[0]),
                    before.sum(dim=0),
                    after.sum(dim=0),
                    before.square().sum(dim=0),
                    after.square().sum(dim=0),
                    (before * after).sum(dim=0),
                ]
            )
        )
    return torch.stack(rows, dim=1)


def lagged_correlation(*, statistics: Tensor) -> Tensor:
    count, first, second, first_square, second_square, product = statistics
    covariance = product / count - first * second / count**2
    deviations = (
        (first_square / count - (first / count) ** 2).clamp_min(1e-12)
        * (second_square / count - (second / count) ** 2).clamp_min(1e-12)
    ).sqrt()
    return covariance / deviations


@dataclass(kw_only=True)
class LeverageError:
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"
    LAGS = 5

    # Lagged sufficient statistics, real then generated.
    statistics: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "LeverageError"

    def __init__(self, *, config: Config | None = None) -> None:
        self.statistics = None

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        lags = min(self.LAGS, target.shape[1] - 1)
        if lags < 1:
            raise ValueError("LeverageError needs windows of at least 2 steps")
        statistics = torch.stack(
            [
                lagged_statistics(windows=windows, lags=lags)
                for windows in (target, prediction)
            ]
        )
        self.statistics = (
            statistics if self.statistics is None else self.statistics + statistics
        )

    def merge(self, *, other: LeverageError) -> None:
        if other.statistics is not None:
            self.statistics = (
                other.statistics
                if self.statistics is None
                else self.statistics + other.statistics
            )

    def compute(self) -> float:
        if self.statistics is None:
            raise ValueError("LeverageError has no observations")
        real, generated = (
            lagged_correlation(statistics=statistics) for statistics in self.statistics
        )
        return float((real - generated).abs().mean())
