"""Parallel convolution kernels for TimesNet."""

import torch
from torch import Tensor, nn


class InceptionBlock(nn.Module):
    def __init__(
        self, *, in_channels: int, out_channels: int, num_inception_kernels: int = 6
    ) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 2 * index + 1, padding=index)
            for index in range(num_inception_kernels)
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, *, features: Tensor) -> Tensor:
        return torch.stack([kernel(features) for kernel in self.kernels], dim=-1).mean(
            dim=-1
        )
