"""CRPS of the channel sum within each joint trajectory."""

from dataclasses import dataclass

from torch import Tensor

from kairots.metrics.SampleCRPS import SampleCRPS


class CRPSSum(SampleCRPS):
    ROLES = ("metric",)
    TASKS = ("SampleForecasting",)
    NEEDS = ("samples", "target")
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config(SampleCRPS.Config):
        name: str = "CRPSSum"

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        super().update(prediction=prediction.sum(dim=-2), target=target.sum(dim=-1))
