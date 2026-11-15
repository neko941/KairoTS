"""TimesNet imputation head: observed-only normalization, time-mark embedding, then
blocks and a projection to channels over the unchanged time axis (TSLib `imputation`).

Input channels are `[values | observed mask | calendar]`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from kairots.models.TimesNet.Body import Body

if TYPE_CHECKING:
    from kairots.models.TimesNet import TimesNet


class Imputation(nn.Module):
    def __init__(self, *, config: TimesNet.Config) -> None:
        super().__init__()
        self.channels = config.input_channels
        self.projection = nn.Linear(config.d_model, config.input_channels, bias=True)

    def forward(self, *, batch_x: Tensor, body: Body) -> Tensor:
        values = batch_x[..., : self.channels]
        mask = batch_x[..., self.channels : 2 * self.channels]
        marks = batch_x[..., 2 * self.channels :]
        count = mask.sum(dim=1, keepdim=True)
        means = (values.sum(dim=1, keepdim=True) / count).detach()
        centered = (values - means).masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            (centered * centered).sum(dim=1, keepdim=True) / count + 1e-5
        )
        stdev = stdev.detach()
        features = body.embed(
            batch_x=centered / stdev, marks=marks if marks.shape[-1] else None
        )
        features = body.transform(features=features)
        return self.projection(features) * stdev + means
