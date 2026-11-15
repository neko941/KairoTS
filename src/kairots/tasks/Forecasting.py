"""Point forecasting task semantics.

Without covariates the model reads the history of every channel and forecasts every
channel. With `target_channels` or `calendar` set, the model input is laid out by role:
channels `[targets | past-observed covariates | calendar]` over `input_length +
output_length` steps. History rows hold every value; horizon rows hold only the
known-future calendar features, and the targets and past-observed covariates there
are NaN, so a value not yet observed at the forecast origin never reaches the model.
"""

from dataclasses import dataclass
from typing import Any

import torch

from kairots.tasks.base import Timed


@dataclass(kw_only=True)
class Forecasting:
    """Prepare history and forecast target tensors."""

    HEAD = "Forecasting"
    RECONSTRUCTS = False
    LOSS = "MSE"
    METRICS = ("MAE",)

    @dataclass(kw_only=True)
    class Config(Timed.Config):
        name: str = "Forecasting"
        # Channels to forecast; empty forecasts every input channel. The other input
        # channels are past-observed covariates (their history only).
        target_channels: tuple[str, ...] = ()
        # Source channel order, filled from `data.source.input_channels`.
        input_channels: tuple[str, ...] = ()

    def __init__(self, *, config: Config) -> None:
        self.config = config
        channels = list(config.input_channels)
        unknown = set(config.target_channels) - set(channels)
        if unknown:
            raise ValueError(
                f"target_channels {sorted(unknown)} are not input channels"
            )
        self.covariates = bool(config.target_channels or config.calendar)
        columns = [channels.index(name) for name in config.target_channels] or list(
            range(len(channels))
        )
        past = [index for index in range(len(channels)) if index not in columns]
        calendar = list(range(len(channels), len(channels) + len(config.calendar)))
        # Target columns of the client values, in prediction order.
        self.target_index: slice | list[int] = (
            columns if config.target_channels else slice(0, len(channels) or None)
        )
        # Model input columns: targets, past-observed covariates, calendar.
        self.order = columns + past + calendar
        self.observed = len(channels)

    def prepare(
        self,
        *,
        batch_x: Any,
        batch_y: Any,
        output_length: int,
        generator: Any = None,
    ) -> tuple[Any, Any]:
        target = batch_y[:, -output_length:, self.target_index]
        if not self.covariates:
            return batch_x, target
        future = batch_y[:, -output_length:, self.order].clone()
        future[..., : self.observed] = torch.nan
        return torch.cat([batch_x[..., self.order], future], dim=1), target
