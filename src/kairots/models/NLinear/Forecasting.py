"""NLinear forecasting head."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class Forecasting(nn.Module):
    """NLinear: subtract the last history step, map over time, add it back.

    Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?"
    AAAI 2023, arXiv:2205.13504.
    Official source: cure-lab/LTSF-Linear, models/NLinear.py.
    Protocol: channel-wise point forecast from a last-step-normalized history window.
    The normalization term is detached, so no gradient flows through it, as in the
    official implementation.
    Adaptations: KairoTS tensor/config naming and batch-first [B,T,C] interface.
    """

    @dataclass(kw_only=True)
    class Config:
        input_length: int = 96
        output_length: int = 24
        input_channels: int = 7
        individual: bool = False

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        if min(config.input_length, config.output_length, config.input_channels) <= 0:
            raise ValueError("NLinear dimensions must be positive")
        self.individual = config.individual
        if config.individual:
            self.linears = nn.ModuleList(
                nn.Linear(config.input_length, config.output_length)
                for _ in range(config.input_channels)
            )
        else:
            self.linear = nn.Linear(config.input_length, config.output_length)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        linear = self.linears[0] if self.individual else self.linear
        if batch_x.ndim != 3 or batch_x.shape[1] != linear.in_features:
            raise ValueError(
                "batch_x must have shape [batch, input_length, input_channels]"
            )
        last = batch_x[:, -1:, :].detach()
        residual = batch_x - last
        if self.individual:
            forecast = torch.stack(
                [
                    self.linears[channel](residual[:, :, channel])
                    for channel in range(residual.shape[2])
                ],
                dim=2,
            )
        else:
            forecast = self.linear(residual.transpose(1, 2)).transpose(1, 2)
        return forecast + last
