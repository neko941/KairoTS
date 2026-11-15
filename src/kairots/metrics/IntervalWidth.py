"""Mergeable mean width `q_last - q_first` of the outer central interval over scored
positions. Test-only: a narrower interval is better only at the same coverage
(`CoverageError`), so it never drives model selection."""

from dataclasses import dataclass

from torch import Tensor

from kairots.metrics.CoverageError import central_interval


@dataclass(kw_only=True)
class IntervalWidth:
    ROLES = ("metric",)
    TASKS = ("ProbabilisticForecasting",)
    NEEDS = ("quantile_prediction", "target")
    DIRECTION = "min"
    PHASES = ("test",)

    total: Tensor | None = None
    count: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "IntervalWidth"

    def __init__(self, *, config: Config | None = None) -> None:
        self.total = None
        self.count = None

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        lower, upper, _ = central_interval(prediction=prediction)
        scored = target.isfinite()
        total = (upper - lower).where(scored, 0.0).sum()
        count = scored.sum()
        self.total = total if self.total is None else self.total + total
        self.count = count if self.count is None else self.count + count

    def merge(self, *, other: IntervalWidth) -> None:
        if other.total is not None:
            self.total = other.total if self.total is None else self.total + other.total
        if other.count is not None:
            self.count = other.count if self.count is None else self.count + other.count

    def compute(self) -> float:
        if self.total is None or self.count is None or not self.count:
            raise ValueError("IntervalWidth has no observations")
        return float((self.total / self.count).item())
