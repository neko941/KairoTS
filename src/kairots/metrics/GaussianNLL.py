"""Gaussian negative log likelihood, including the normalizing constant."""

from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.metrics._mean import add, mean, merge


def nll_sum(*, prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Summed density loss and count; final prediction axis is (mean, scale)."""
    if prediction.shape != (*target.shape, 2):
        raise ValueError("Gaussian prediction needs target shape plus (mean, scale)")
    distribution = torch.distributions.Normal(
        loc=prediction[..., 0], scale=prediction[..., 1], validate_args=True
    )
    scored = target.isfinite()
    losses = -distribution.log_prob(torch.where(scored, target, 0.0))
    return torch.where(scored, losses, 0.0).sum(), scored.sum()


class GaussianNLL:
    ROLES = ("loss", "metric")
    TASKS = ("LikelihoodForecasting",)
    NEEDS = ("gaussian_prediction", "target")
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config:
        name: str = "GaussianNLL"

    def __init__(self, *, config: Config | None = None) -> None:
        self.total: Tensor | None = None
        self.count: Tensor | None = None

    def __call__(self, *, prediction: Tensor, target: Tensor) -> Tensor:
        total, count = nll_sum(prediction=prediction, target=target)
        if not count:
            raise ValueError("GaussianNLL has no scored targets")
        return total / count

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        total, count = nll_sum(prediction=prediction, target=target)
        add(metric=self, total=total, count=count)

    def merge(self, *, other: "GaussianNLL") -> None:
        merge(metric=self, other=other)

    def compute(self) -> float:
        return mean(metric=self)
