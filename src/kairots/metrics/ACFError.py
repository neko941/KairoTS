"""Mergeable autocorrelation error: the mean absolute difference between the real and
generated autocorrelation at lags 1 to `LAGS` (fewer when windows are shorter), each
the mean over windows of a window's per-channel autocorrelation, averaged over lags
and channels. A constant window has autocorrelation 0. Merging adds the sums.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def window_acf(*, windows: Tensor, lags: int) -> Tensor:
    """(windows, lags, channels): each window's autocorrelation at lags 1 to `lags`."""
    centered = windows.double() - windows.double().mean(dim=1, keepdim=True)
    variance = centered.square().sum(dim=1).clamp_min(1e-12)
    return torch.stack(
        [
            (centered[:, lag:] * centered[:, :-lag]).sum(dim=1) / variance
            for lag in range(1, lags + 1)
        ],
        dim=1,
    )


@dataclass(kw_only=True)
class ACFError:
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"
    LAGS = 10

    real: Tensor | None = None
    generated: Tensor | None = None
    count: int = 0

    @dataclass(kw_only=True)
    class Config:
        name: str = "ACFError"

    def __init__(self, *, config: Config | None = None) -> None:
        self.real = None
        self.generated = None
        self.count = 0

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("ACFError needs as many generated as real windows")
        lags = min(self.LAGS, target.shape[1] - 1)
        if lags < 1:
            raise ValueError("ACFError needs windows of at least 2 steps")
        real = window_acf(windows=target, lags=lags).sum(dim=0)
        generated = window_acf(windows=prediction, lags=lags).sum(dim=0)
        self.real = real if self.real is None else self.real + real
        self.generated = (
            generated if self.generated is None else self.generated + generated
        )
        self.count += target.shape[0]

    def merge(self, *, other: ACFError) -> None:
        if other.real is None or other.generated is None:
            return
        self.real = other.real if self.real is None else self.real + other.real
        self.generated = (
            other.generated
            if self.generated is None
            else self.generated + other.generated
        )
        self.count += other.count

    def compute(self) -> float:
        if self.real is None or self.generated is None or not self.count:
            raise ValueError("ACFError has no observations")
        return float(((self.real - self.generated) / self.count).abs().mean())
