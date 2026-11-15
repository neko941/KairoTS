"""Absolute error of the sample median at scored positions."""

from dataclasses import dataclass

from torch import Tensor

from kairots.metrics.MAE import MAE


class MedianMAE(MAE):
    ROLES = ("metric",)
    TASKS = ("ProbabilisticImputation",)
    NEEDS = ("samples", "target")
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config(MAE.Config):
        name: str = "MedianMAE"

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        median_index = (prediction.shape[-1] - 1) // 2
        median = prediction.sort(dim=-1).values[..., median_index]
        super().update(prediction=median, target=target)
