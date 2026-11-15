"""Pinball (quantile) loss over scored positions and evenly spaced levels.

The prediction is `[..., Q]` against a `[...]` target; level `k` is
`(k + 1) / (Q + 1)` (see `tasks/ProbabilisticForecasting.py`). A non-finite target
marks an unscored position (see `metrics/MSE.py`).
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def pinball_sum(*, prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Sum of pinball losses over scored positions and levels, and the scored count."""
    if prediction.shape[:-1] != target.shape:
        raise ValueError(
            f"quantile prediction {tuple(prediction.shape)} must be the target shape "
            f"{tuple(target.shape)} plus a quantile axis"
        )
    num_quantiles = prediction.shape[-1]
    levels = torch.arange(1, num_quantiles + 1, device=prediction.device) / (
        num_quantiles + 1
    )
    scored = target.isfinite()
    error = torch.where(scored, target, 0.0).unsqueeze(-1) - prediction
    loss = torch.maximum(levels * error, (levels - 1) * error)
    return torch.where(scored.unsqueeze(-1), loss, 0.0).sum(), scored.sum()


class Pinball:
    ROLES = ("loss",)
    TASKS = ("ProbabilisticForecasting",)
    NEEDS = ("quantile_prediction", "target")

    @dataclass(kw_only=True)
    class Config:
        name: str = "Pinball"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def __call__(self, *, prediction: Tensor, target: Tensor) -> Tensor:
        total, count = pinball_sum(prediction=prediction, target=target)
        return total / (count * prediction.shape[-1])
