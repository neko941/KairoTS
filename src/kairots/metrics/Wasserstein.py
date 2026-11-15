"""Mergeable 1-Wasserstein distance between the real and generated marginal
distribution of each channel (every value of every window), averaged over channels:
the mean absolute difference of the sorted values, so both sides hold the same
number of windows. Merging keeps the windows.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(kw_only=True)
class Wasserstein:
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"

    real: list[Tensor] = field(default_factory=list)
    generated: list[Tensor] = field(default_factory=list)

    @dataclass(kw_only=True)
    class Config:
        name: str = "Wasserstein"

    def __init__(self, *, config: Config | None = None) -> None:
        self.real = []
        self.generated = []

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("Wasserstein needs as many generated as real windows")
        self.generated.append(prediction.flatten(0, 1).double())
        self.real.append(target.flatten(0, 1).double())

    def merge(self, *, other: Wasserstein) -> None:
        self.real += other.real
        self.generated += other.generated

    def compute(self) -> float:
        if not self.real:
            raise ValueError("Wasserstein has no observations")
        real = torch.cat(self.real).sort(dim=0).values
        generated = torch.cat(self.generated).sort(dim=0).values
        return float((real - generated).abs().mean())
