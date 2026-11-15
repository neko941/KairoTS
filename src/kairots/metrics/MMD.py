"""Mergeable maximum mean discrepancy (squared, biased V-statistic) between real and
generated windows, each flattened to one vector of `d` values, with the RBF kernel
`exp(-|x - y|^2 / d)` (scikit-learn's default gamma `1 / d`), as in TimeDP's
`mmd_metric`. Merging keeps the windows.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(kw_only=True)
class MMD:
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"

    real: list[Tensor] = field(default_factory=list)
    generated: list[Tensor] = field(default_factory=list)

    @dataclass(kw_only=True)
    class Config:
        name: str = "MMD"

    def __init__(self, *, config: Config | None = None) -> None:
        self.real = []
        self.generated = []

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        self.generated.append(prediction.flatten(1).double())
        self.real.append(target.flatten(1).double())

    def merge(self, *, other: MMD) -> None:
        self.real += other.real
        self.generated += other.generated

    def compute(self) -> float:
        if not self.real:
            raise ValueError("MMD has no observations")
        real, generated = torch.cat(self.real), torch.cat(self.generated)
        gamma = 1 / real.shape[1]

        def kernel_mean(*, left: Tensor, right: Tensor) -> Tensor:
            return torch.exp(-gamma * torch.cdist(left, right).square()).mean()

        return float(
            kernel_mean(left=real, right=real)
            + kernel_mean(left=generated, right=generated)
            - 2 * kernel_mean(left=real, right=generated)
        )
