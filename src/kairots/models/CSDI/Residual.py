"""Residual used by CSDI."""

import math

from torch import Tensor, nn

from kairots.models.CSDI.convolution import convolution


class Residual(nn.Module):
    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.step = nn.Linear(128, width)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=64,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.feature = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=64,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.middle = convolution(source=width, destination=2 * width)
        self.side = convolution(source=145, destination=2 * width)
        self.output = convolution(source=width, destination=2 * width)

    def forward(
        self, *, features: Tensor, side: Tensor, step: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, width, channels, length = features.shape
        hidden = features + self.step(step)[:, :, None, None]
        if length > 1:
            hidden = self.temporal(
                hidden.permute(0, 2, 3, 1).reshape(-1, length, width)
            )
            hidden = hidden.reshape(batch, channels, length, width).permute(0, 3, 1, 2)
        if channels > 1:
            hidden = self.feature(
                hidden.permute(0, 3, 2, 1).reshape(-1, channels, width)
            )
            hidden = hidden.reshape(batch, length, channels, width).permute(0, 3, 2, 1)
        hidden = self.middle(hidden.reshape(batch, width, -1)) + self.side(
            side.flatten(2)
        )
        gate, filtered = hidden.chunk(2, dim=1)
        residual, skip = self.output(gate.sigmoid() * filtered.tanh()).chunk(2, dim=1)
        return (features + residual.reshape_as(features)) / math.sqrt(
            2
        ), skip.reshape_as(features)
