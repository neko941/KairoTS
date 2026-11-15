"""Decompose a series into seasonal and trend components."""

import torch
from torch import Tensor, nn


class SeriesDecomposition(nn.Module):
    """Autoformer series_decomp: trend = edge-padded moving average, seasonal = rest."""

    def __init__(self, *, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("moving_average_kernel must be a positive odd kernel size")
        self.pad = (kernel_size - 1) // 2
        self.average = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, *, batch_x: Tensor) -> tuple[Tensor, Tensor]:
        front = batch_x[:, :1, :].expand(-1, self.pad, -1)
        end = batch_x[:, -1:, :].expand(-1, self.pad, -1)
        padded = torch.cat([front, batch_x, end], dim=1)
        trend = self.average(padded.transpose(1, 2)).transpose(1, 2)
        return batch_x - trend, trend
