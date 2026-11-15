"""Gaussian likelihood head: a mean and positive standard deviation per position."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from kairots.models.DLinear.Quantile import Quantile

if TYPE_CHECKING:
    from kairots.models.DLinear import DLinear


class Gaussian(Quantile):
    def __init__(self, *, config: DLinear.Config) -> None:
        super().__init__(config=replace(config, num_quantiles=2))

    def forward(self, *, batch_x: Tensor, body: nn.Module) -> Tensor:
        values = super().forward(batch_x=batch_x, body=body)
        scale = softplus(values[..., 1]).clamp_min(torch.finfo(values.dtype).eps)
        return torch.stack([values[..., 0], scale], dim=-1)
