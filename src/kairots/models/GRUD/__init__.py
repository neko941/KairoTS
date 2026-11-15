"""GRU-D classification for multivariate time series with missing values.

Che et al., "Recurrent Neural Networks for Multivariate Time Series with
Missing Values", Scientific Reports 8, 6085 (2018). Inputs are
standardized panels `[B, input_length, input_channels]`; channel 0 is time and
remaining channels are measurements with NaN for missing values. Trailing rows
with NaN in every channel are padding and do not update the hidden state.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kairots.models.base import Dropout, HiddenWidth


class GRUD(nn.Module):
    """Classify irregular panels with feature and hidden-state decay."""

    TASKS = ("Classification",)
    # Reads NaN as unobserved (masks and time since last observation).
    MASKS = True

    @dataclass(kw_only=True)
    class Config(HiddenWidth.Config, Dropout.Config):
        name: str = "GRUD"
        input_length: int = 203
        input_channels: int = 41
        num_classes: int = 2
        dropout: float = 0.3

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        if config.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if config.input_channels < 2:
            raise ValueError("input_channels must be at least 2")
        self.config = config
        variables = config.input_channels - 1
        self.feature_decay_weight = nn.Parameter(torch.ones(variables))
        self.feature_decay_bias = nn.Parameter(torch.zeros(variables))
        self.hidden_decay = nn.Linear(in_features=variables, out_features=config.hidden)
        self.cell = nn.GRUCell(2 * variables, config.hidden)
        self.dropout = nn.Dropout(p=config.dropout)
        self.projection = nn.Linear(
            in_features=config.hidden, out_features=config.num_classes
        )

    def forward(self, *, batch_x: Tensor) -> Tensor:
        values = batch_x[:, :, 1:]
        observed = ~torch.isnan(values)
        valid_row = ~torch.isnan(batch_x[:, :, 0])
        time = batch_x[:, :, 0]
        batch_size = batch_x.shape[0]
        variables = values.shape[2]
        hidden = values.new_zeros((batch_size, self.config.hidden))
        last_value = values.new_zeros((batch_size, variables))
        delta = values.new_zeros((batch_size, variables))
        previous_time = values.new_zeros((batch_size,))

        for index in range(batch_x.shape[1]):
            row_valid = valid_row[:, index]
            mask = observed[:, index]
            current_time = time[:, index]
            if index == 0:
                row_delta = torch.zeros_like(delta)
            else:
                row_delta = (
                    current_time[:, None]
                    - previous_time[:, None]
                    + (~observed[:, index - 1]).to(values.dtype) * delta
                )
            row_delta = torch.where(row_valid[:, None], row_delta, delta)
            gamma_x = torch.exp(
                -torch.relu(
                    self.feature_decay_weight * row_delta + self.feature_decay_bias
                )
            )
            gamma_h = torch.exp(-torch.relu(self.hidden_decay(row_delta)))
            safe_value = torch.where(
                mask, values[:, index], torch.zeros_like(last_value)
            )
            estimate = torch.where(mask, safe_value, gamma_x * last_value)
            proposed = self.cell(
                torch.cat((estimate, mask.to(values.dtype)), dim=1),
                hidden * gamma_h,
            )
            hidden = torch.where(row_valid[:, None], proposed, hidden)
            last_value = torch.where(mask, safe_value, last_value)
            delta = row_delta
            previous_time = torch.where(row_valid, current_time, previous_time)

        return self.projection(self.dropout(hidden))
