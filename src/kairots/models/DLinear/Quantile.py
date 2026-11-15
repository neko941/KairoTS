"""DLinear quantile head: project each channel's horizon to every quantile level."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from kairots.models.DLinear.Forecasting import Forecasting

if TYPE_CHECKING:
    from kairots.models.DLinear import DLinear


class Quantile(nn.Module):
    """`[B, H, C]` body output to `[B, H, C, Q]` quantiles, shared over channels.

    No monotonicity constraint, so levels may cross; sort or use increments
    if a method needs ordered quantiles. Moves to `models/heads/` when a second model
    reuses it.
    """

    # The body maps input_length to output_length, as for forecasting.
    body = Forecasting.body

    def __init__(self, *, config: DLinear.Config) -> None:
        super().__init__()
        self.output_length = config.output_length
        self.num_quantiles = config.num_quantiles
        self.projection = nn.Linear(
            config.output_length, config.output_length * config.num_quantiles
        )

    def forward(self, *, batch_x: Tensor, body: nn.Module) -> Tensor:
        forecast = body(batch_x=batch_x)[:, -self.output_length :, :].transpose(1, 2)
        quantiles = self.projection(forecast).unflatten(
            -1, (self.output_length, self.num_quantiles)
        )
        return quantiles.transpose(1, 2)
