"""DLinear body: moving-average decomposition plus seasonal and trend linears."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kairots.models.DLinear.SeriesDecomposition import SeriesDecomposition


class Body(nn.Module):
    """DLinear: seasonal and trend parts each map history to the forecast.

    Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?"
    AAAI 2023, arXiv:2205.13504.
    Original source: cure-lab/LTSF-Linear, models/DLinear.py.
    Protocol: channel-independent linear maps on the decomposed history window.
    Adaptations: KairoTS names (seq_len -> input_length, pred_len -> output_length,
    enc_in -> input_channels), batch-first [B, T, C] in and out, task branches moved
    to heads. Parameters keep the original nn.Linear initialization. The paper's
    weight-visualization experiment instead starts weights at `1 / input_length`;
    `Average` applies that initialization through body-weight patterns.
    """

    @dataclass(kw_only=True)
    class Config:
        input_length: int = 96
        output_length: int = 24
        input_channels: int = 7
        moving_average_kernel: int = 25
        individual: bool = False

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        if min(config.input_length, config.output_length, config.input_channels) <= 0:
            raise ValueError("DLinear dimensions must be positive")
        self.decomposition = SeriesDecomposition(
            kernel_size=config.moving_average_kernel
        )
        self.individual = config.individual
        count = config.input_channels if config.individual else 1
        self.linear_seasonal = nn.ModuleList(
            self._linear(
                input_length=config.input_length, output_length=config.output_length
            )
            for _ in range(count)
        )
        self.linear_trend = nn.ModuleList(
            self._linear(
                input_length=config.input_length, output_length=config.output_length
            )
            for _ in range(count)
        )

    @staticmethod
    def _linear(*, input_length: int, output_length: int) -> nn.Linear:
        return nn.Linear(in_features=input_length, out_features=output_length)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        seasonal, trend = self.decomposition(batch_x=batch_x)
        seasonal, trend = seasonal.transpose(1, 2), trend.transpose(1, 2)
        if self.individual:
            output = torch.stack(
                [
                    self.linear_seasonal[channel](seasonal[:, channel, :])
                    + self.linear_trend[channel](trend[:, channel, :])
                    for channel in range(seasonal.shape[1])
                ],
                dim=1,
            )
        else:
            output = self.linear_seasonal[0](seasonal) + self.linear_trend[0](trend)
        return output.transpose(1, 2)
