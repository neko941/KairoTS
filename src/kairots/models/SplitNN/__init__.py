"""SplitNN for vertically partitioned forecasting (Vepakomma et al., "Split
learning for health", arXiv:1812.00564, 2018).

The input channels are owned by parties. The label party owns the target channels
(their history and their future values); each feature party owns a contiguous
block of the past-observed covariates, sized by `party_channels` in the order of
`data.source.input_channels` without the targets. Every party has a bottom model
that maps only its own history to an `embedding_dimension` cut-layer activation. The
label party's top model reads the concatenated activations and emits the forecast,
plus a linear map of its own target history.

Input layout (`tasks/Forecasting.py` with `target_channels`): `[targets | past]`
channels over `input_length + output_length` steps; only history rows are read,
so the NaN horizon rows never reach a bottom model.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kairots.models.base import HiddenWidth


class SplitNN(nn.Module):
    TASKS = ("Forecasting",)
    COVARIATES = ("past",)
    STRATEGIES = ("SplitNN",)

    @dataclass(kw_only=True)
    class Config(HiddenWidth.Config):
        name: str = "SplitNN"
        input_length: int = 96
        output_length: int = 24
        input_channels: int = 7
        output_channels: int = 1
        calendar_channels: int = 0
        # Covariate channels per feature party; their sum is input_channels minus
        # output_channels. Empty: the label party alone.
        party_channels: tuple[int, ...] = (3, 3)
        embedding_dimension: int = 64
        hidden: int = 128

        def __post_init__(self) -> None:
            self.party_channels = tuple(self.party_channels)
            if self.calendar_channels:
                raise ValueError("SplitNN takes no calendar features")
            if sum(self.party_channels) != self.input_channels - self.output_channels:
                raise ValueError(
                    f"party_channels {self.party_channels} must sum to the "
                    f"{self.input_channels - self.output_channels} covariate channels"
                )

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.config = config
        widths = (config.output_channels, *config.party_channels)
        self.bounds = [sum(widths[:index]) for index in range(len(widths) + 1)]
        # bottoms[0] belongs to the label party, bottoms[k] to feature party k.
        self.bottoms = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.input_length * width, config.embedding_dimension),
                nn.ReLU(),
            )
            for width in widths
        )
        self.top = nn.Sequential(
            nn.Linear(len(widths) * config.embedding_dimension, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.output_length * config.output_channels),
        )
        self.residual = nn.Linear(config.input_length, config.output_length)

    def activations(self, *, batch_x: Tensor) -> list[Tensor]:
        """Cut-layer activation of each party, label party first."""
        history = batch_x[:, : self.config.input_length]
        return [
            bottom(history[..., start:end].flatten(start_dim=1))
            for bottom, start, end in zip(
                self.bottoms, self.bounds[:-1], self.bounds[1:], strict=True
            )
        ]

    def forward(self, *, batch_x: Tensor) -> Tensor:
        config = self.config
        own = batch_x[:, : config.input_length, : config.output_channels]
        joint = self.top(torch.cat(self.activations(batch_x=batch_x), dim=1))
        forecast = joint.view(-1, config.output_length, config.output_channels)
        return forecast + self.residual(own.transpose(1, 2)).transpose(1, 2)
