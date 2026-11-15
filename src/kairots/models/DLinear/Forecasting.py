"""DLinear forecasting head: the body maps history to the forecast."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from kairots.models.DLinear.Body import Body

if TYPE_CHECKING:
    from kairots.models.DLinear import DLinear


class Forecasting(nn.Module):
    """Keep the last output_length body steps, as in the official code."""

    @staticmethod
    def body(*, config: DLinear.Config) -> Body.Config:
        return Body.Config(
            input_length=config.input_length,
            output_length=config.output_length,
            input_channels=config.input_channels,
            moving_average_kernel=config.moving_average_kernel,
            individual=config.individual,
        )

    def __init__(self, *, config: DLinear.Config) -> None:
        super().__init__()
        self.output_length = config.output_length

    def forward(self, *, batch_x: Tensor, body: nn.Module) -> Tensor:
        return body(batch_x=batch_x)[:, -self.output_length :, :]
