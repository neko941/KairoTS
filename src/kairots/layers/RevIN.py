"""Reversible instance normalization (Kim et al., ICLR 2022), per window and channel.

Optional learnable per-channel affine (`affine`) and a detached scale
(`detach_scale`), as in the official RevIN and its ports (TimeKAN's
`StandardNorm`); iTransformer uses neither (no affine, gradient through the scale).
"""

from typing import Literal

import torch
from torch import Tensor, nn


class RevIN(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        epsilon: float = 1e-5,
        affine: bool = False,
        detach_scale: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.epsilon = epsilon
        self.detach_scale = detach_scale
        self.weight: nn.Parameter | None = None
        self.bias: nn.Parameter | None = None
        if affine:
            self.weight = nn.Parameter(torch.ones(input_channels))
            self.bias = nn.Parameter(torch.zeros(input_channels))
        self.mean: Tensor | None = None
        self.stdev: Tensor | None = None

    def forward(
        self,
        *,
        features: Tensor,
        mode: Literal["norm", "denorm"],
        output_channels: int | None = None,
    ) -> Tensor:
        if mode == "norm":
            if features.shape[-1] != self.input_channels:
                raise ValueError("RevIN feature count does not match its configuration")
            self.mean = features.mean(dim=1, keepdim=True).detach()
            centered = features - self.mean
            stdev = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + self.epsilon
            )
            self.stdev = stdev.detach() if self.detach_scale else stdev
            normalized = centered / self.stdev
            if self.weight is not None and self.bias is not None:
                normalized = normalized * self.weight + self.bias
            return normalized
        if mode != "denorm":
            raise ValueError(f"Unknown RevIN mode: {mode}")
        if self.mean is None or self.stdev is None:
            raise RuntimeError("RevIN denorm requires a preceding norm call")
        channels = features.shape[-1] if output_channels is None else output_channels
        if self.weight is not None and self.bias is not None:
            features = (features - self.bias[:channels]) / (
                self.weight[:channels] + self.epsilon**2
            )
        return features * self.stdev[..., :channels] + self.mean[..., :channels]
