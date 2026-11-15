"""Empirical CRPS with exact pairwise sample term computed by sorting."""

from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.metrics._mean import add, mean, merge


def sample_crps(*, prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if prediction.shape[:-1] != target.shape or prediction.shape[-1] < 1:
        raise ValueError("sample predictions need target shape plus a sample axis")
    mask = target.isfinite()
    samples = prediction.shape[-1]
    ordered = prediction.sort(dim=-1).values
    weights = 2 * torch.arange(1, samples + 1, device=prediction.device) - samples - 1
    score = (prediction - target.nan_to_num().unsqueeze(-1)).abs().mean(dim=-1)
    score = score - (ordered * weights).sum(dim=-1) / samples**2
    return torch.where(mask, score, 0.0).sum(), mask.sum()


class SampleCRPS:
    ROLES = ("loss", "metric")
    TASKS = ("SampleForecasting", "ProbabilisticImputation")
    NEEDS = ("samples", "target")
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config:
        name: str = "SampleCRPS"

    def __init__(self, *, config: Config | None = None) -> None:
        self.total: Tensor | None = None
        self.count: Tensor | None = None

    def __call__(self, *, prediction: Tensor, target: Tensor) -> Tensor:
        total, count = sample_crps(prediction=prediction, target=target)
        return total / count.clamp_min(1)

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        total, count = sample_crps(prediction=prediction, target=target)
        add(metric=self, total=total, count=count)

    def merge(self, *, other: "SampleCRPS") -> None:
        merge(metric=self, other=other)

    def compute(self) -> float:
        return mean(metric=self)
