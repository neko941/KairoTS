"""Mergeable tail-risk check: the mean over channels of the absolute difference
between the real and generated Value-at-Risk, the `LEVEL` quantile of every value of
every window (the lower tail: losses of returns). Merging keeps the values.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


def tail_risk(*, values: Tensor, level: float) -> tuple[Tensor, Tensor]:
    """Per-channel Value-at-Risk (the `level` quantile) and Expected Shortfall (the
    mean of the values at or below it) of `values`, (rows, channels)."""
    value_at_risk = torch.quantile(values, level, dim=0)
    tail = values <= value_at_risk
    shortfall = torch.where(tail, values, 0.0).sum(dim=0) / tail.sum(dim=0)
    return value_at_risk, shortfall


@dataclass(kw_only=True)
class VaRError:
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"
    LEVEL = 0.05

    real: list[Tensor] = field(default_factory=list)
    generated: list[Tensor] = field(default_factory=list)

    @dataclass(kw_only=True)
    class Config:
        name: str = "VaRError"

    def __init__(self, *, config: Config | None = None) -> None:
        self.real = []
        self.generated = []

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        self.generated.append(prediction.flatten(0, 1).double())
        self.real.append(target.flatten(0, 1).double())

    def merge(self, *, other: VaRError) -> None:
        self.real += other.real
        self.generated += other.generated

    def _risks(self) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if not self.real:
            raise ValueError(f"{type(self).__name__} has no observations")
        return (
            tail_risk(values=torch.cat(self.real), level=self.LEVEL),
            tail_risk(values=torch.cat(self.generated), level=self.LEVEL),
        )

    def compute(self) -> float:
        (real, _), (generated, _) = self._risks()
        return float((real - generated).abs().mean())
