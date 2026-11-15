"""Mergeable heavy-tail check: the mean over channels of the absolute difference
between the real and generated excess kurtosis `E[(x - mu)^4] / sigma^4 - 3`, each
over every value of every window (a stylized fact of returns). Merging adds the
power sums.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def excess_kurtosis(*, sums: Tensor) -> Tensor:
    """Per-channel excess kurtosis from power sums (count, sum x, x^2, x^3, x^4)."""
    _, first, second, third, fourth = sums / sums[0]
    variance = (second - first**2).clamp_min(1e-12)
    central = fourth - 4 * first * third + 6 * first**2 * second - 3 * first**4
    return central / variance**2 - 3


@dataclass(kw_only=True)
class KurtosisError:
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"

    # Power sums (count, sum x, ..., sum x^4) per channel, real then generated.
    sums: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "KurtosisError"

    def __init__(self, *, config: Config | None = None) -> None:
        self.sums = None

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        sums = torch.stack(
            [
                torch.stack(
                    [values.flatten(0, 1).double() ** power for power in range(5)]
                ).sum(dim=1)
                for values in (target, prediction)
            ]
        )
        self.sums = sums if self.sums is None else self.sums + sums

    def merge(self, *, other: KurtosisError) -> None:
        if other.sums is not None:
            self.sums = other.sums if self.sums is None else self.sums + other.sums

    def compute(self) -> float:
        if self.sums is None:
            raise ValueError("KurtosisError has no observations")
        real, generated = (excess_kurtosis(sums=sums) for sums in self.sums)
        return float((real - generated).abs().mean())
