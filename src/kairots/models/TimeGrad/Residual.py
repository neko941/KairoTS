"""Residual used by TimeGrad."""

import math

from torch import Tensor, nn
from torch.nn import functional


class Residual(nn.Module):
    def __init__(self, *, width: int, dilation: int) -> None:
        super().__init__()
        self.step = nn.Linear(64, width)
        self.condition = nn.Conv1d(1, 2 * width, kernel_size=1)
        self.dilated = nn.Conv1d(
            width,
            2 * width,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            padding_mode="circular",
        )
        self.output = nn.Conv1d(width, 2 * width, kernel_size=1)
        for layer in (self.condition, self.output):
            nn.init.kaiming_normal_(layer.weight)

    def forward(
        self, *, features: Tensor, condition: Tensor, step: Tensor
    ) -> tuple[Tensor, Tensor]:
        hidden = self.dilated(
            features + self.step(step).unsqueeze(-1)
        ) + self.condition(condition)
        gate, filtered = hidden.chunk(2, dim=1)
        residual, skip = functional.relu(
            self.output(gate.sigmoid() * filtered.tanh())
        ).chunk(2, dim=1)
        return (features + residual) / math.sqrt(2), skip
