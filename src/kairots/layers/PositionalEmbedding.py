"""Sinusoidal positional embedding (Vaswani et al., 2017)."""

import math

import torch
from torch import Tensor, nn


class PositionalEmbedding(nn.Module):
    def __init__(self, *, d_model: int, max_length: int = 5000) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scales = (
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        ).exp()
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales)
        self.pe: Tensor
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, *, features: Tensor) -> Tensor:
        return self.pe[:, : features.size(1)]
