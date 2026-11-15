"""Mergeable volatility-clustering check: `ACFError` on absolute values, the mean
absolute difference between the real and generated autocorrelation of `|x|` at lags
1 to `LAGS` within windows (a stylized fact of returns: large moves follow large
moves).
"""

from dataclasses import dataclass

from torch import Tensor

from kairots.metrics.ACFError import ACFError


class VolatilityClusteringError(ACFError):
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config(ACFError.Config):
        name: str = "VolatilityClusteringError"

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        super().update(prediction=prediction.abs(), target=target.abs())
