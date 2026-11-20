"""LTSF-Linear forecasting head."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class Forecasting(nn.Module):
    """LTSF-Linear: per-channel nn.Linear(input_length, output_length) over time.

    Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?"
    AAAI 2023, arXiv:2205.13504.
    Official source: cure-lab/LTSF-Linear, models/Linear.py.
    Protocol: independent channel-wise point forecast from a fixed history window.
    Adaptations: KairoTS tensor/config naming and batch-first [B,T,C] interface.
    See README.ai.md for notation and validation coverage.
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
            raise ValueError("Linear dimensions must be positive")
        self.individual = config.individual
        if config.individual:
            # One temporal map per variate, as the official `individual` option.
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
        if self.individual:
            return torch.stack(
                [
                    self.linears[channel](batch_x[:, :, channel])
                    for channel in range(batch_x.shape[2])
                ],
                dim=2,
            )
        return self.linear(batch_x.transpose(1, 2)).transpose(1, 2)
