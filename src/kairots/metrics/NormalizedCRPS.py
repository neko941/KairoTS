"""CSDI's normalized CRPS using nineteen sample quantiles."""

from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.metrics.CRPS import CRPS
from kairots.metrics.Pinball import pinball_sum


class NormalizedCRPS(CRPS):
    ROLES = ("metric",)
    TASKS = ("ProbabilisticImputation",)
    NEEDS = ("samples", "target")
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config(CRPS.Config):
        name: str = "NormalizedCRPS"

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        levels = (
            torch.arange(1, 20, device=prediction.device, dtype=prediction.dtype) / 20
        )
        quantiles = torch.quantile(prediction, q=levels, dim=-1).movedim(0, -1)
        total, _ = pinball_sum(prediction=quantiles, target=target)
        total = 2 * total / 19
        denominator = torch.where(target.isfinite(), target, 0.0).abs().sum()
        self.total = total if self.total is None else self.total + total
        self.count = denominator if self.count is None else self.count + denominator
