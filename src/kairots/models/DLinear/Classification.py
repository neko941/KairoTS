"""DLinear classification head: flatten the body features and project to classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from kairots.models.DLinear.Body import Body

if TYPE_CHECKING:
    from kairots.models.DLinear import DLinear


class Classification(nn.Module):
    """Official DLinear: keep all history steps, then a flattened linear head."""

    @staticmethod
    def body(*, config: DLinear.Config) -> Body.Config:
        return Body.Config(
            input_length=config.input_length,
            output_length=config.input_length,
            input_channels=config.input_channels,
            moving_average_kernel=config.moving_average_kernel,
            individual=config.individual,
        )

    def __init__(self, *, config: DLinear.Config) -> None:
        super().__init__()
        if config.num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        self.projection = nn.Linear(
            in_features=config.input_channels * config.input_length,
            out_features=config.num_classes,
        )

    def forward(self, *, batch_x: Tensor, body: nn.Module) -> Tensor:
        return self.projection(body(batch_x=batch_x).flatten(start_dim=1))
