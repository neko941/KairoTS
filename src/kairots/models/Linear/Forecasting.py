"""LTSF-Linear forecasting head."""

from dataclasses import dataclass

from torch import Tensor, nn


class Forecasting(nn.Module):
    """LTSF-Linear: per-channel nn.Linear(input_length, output_length) over time.

    Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?"
    AAAI 2023, arXiv:2205.13504.
    Official source: cure-lab/LTSF-Linear, Models/Linear.py.
    Protocol: independent channel-wise point forecast from a fixed history window.
    Adaptations: KairoTS tensor/config naming and batch-first [B,T,C] interface.
    See README.ai.md for notation and validation coverage.
    """

    @dataclass(kw_only=True)
    class Config:
        input_length: int = 96
        output_length: int = 24
        input_channels: int = 7

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        if min(config.input_length, config.output_length, config.input_channels) <= 0:
            raise ValueError("Linear dimensions must be positive")
        self.linear = nn.Linear(config.input_length, config.output_length)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        if batch_x.ndim != 3 or batch_x.shape[1] != self.linear.in_features:
            raise ValueError(
                "batch_x must have shape [batch, input_length, input_channels]"
            )
        return self.linear(batch_x.transpose(1, 2)).transpose(1, 2)
