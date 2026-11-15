"""Mergeable CRPS approximated from quantiles: twice the pinball loss averaged over
the evenly spaced levels, pooled over scored positions (Gneiting and Raftery, 2007,
quantile decomposition). Marginal per position; it says nothing about joint
trajectories.
"""

from dataclasses import dataclass

from torch import Tensor

from kairots.metrics.Pinball import pinball_sum


@dataclass(kw_only=True)
class CRPS:
    ROLES = ("metric",)
    TASKS = ("ProbabilisticForecasting",)
    NEEDS = ("quantile_prediction", "target")
    DIRECTION = "min"

    total: Tensor | None = None
    count: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "CRPS"

    def __init__(self, *, config: Config | None = None) -> None:
        self.total = None
        self.count = None

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        total, count = pinball_sum(prediction=prediction, target=target)
        value = 2 * total / prediction.shape[-1]
        self.total = value if self.total is None else self.total + value
        self.count = count if self.count is None else self.count + count

    def merge(self, *, other: CRPS) -> None:
        if other.total is not None:
            self.total = other.total if self.total is None else self.total + other.total
        if other.count is not None:
            self.count = other.count if self.count is None else self.count + other.count

    def compute(self) -> float:
        if self.total is None or self.count is None or not self.count:
            raise ValueError("CRPS has no observations")
        return float((self.total / self.count).item())
