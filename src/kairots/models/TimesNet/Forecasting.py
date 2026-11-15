"""TimesNet forecasting head (also reconstruction): per-window normalization, a time
extension from history to history plus forecast between embedding and blocks,
then a projection to channels keeping the last output_length steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from kairots.models.TimesNet.Body import Body

if TYPE_CHECKING:
    from kairots.models.TimesNet import TimesNet


class Forecasting(nn.Module):
    def __init__(self, *, config: TimesNet.Config) -> None:
        super().__init__()
        self.output_length = config.output_length
        # Reconstruction keeps the time axis; forecasting extends it at any length.
        self.predict_linear = (
            nn.Linear(config.input_length, config.input_length + config.output_length)
            if not config.reconstruct
            else None
        )
        self.projection = nn.Linear(config.d_model, config.input_channels, bias=True)

    def forward(self, *, batch_x: Tensor, body: Body) -> Tensor:
        # TSLib's per-window normalization. Imputation gets no mask, so the
        # statistics include the zeroed hidden positions (TSLib `imputation` uses
        # observed ones only).
        means = batch_x.mean(dim=1, keepdim=True).detach()
        centered = batch_x - means
        stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        features = body.embed(batch_x=centered / stdev)
        if self.predict_linear is not None:
            features = self.predict_linear(features.permute(0, 2, 1)).permute(0, 2, 1)
        features = body.transform(features=features)
        forecast = self.projection(features)[:, -self.output_length :, :]
        return forecast * stdev + means
