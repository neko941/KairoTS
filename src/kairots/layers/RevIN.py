"""Per-window reversible instance normalization used by official experiments."""

from typing import Literal

import torch
from torch import Tensor, nn


class RevIN(nn.Module):
    def __init__(self, *, input_channels: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.epsilon = epsilon
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
            self.stdev = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + self.epsilon
            )
            return centered / self.stdev
        if mode != "denorm":
            raise ValueError(f"Unknown RevIN mode: {mode}")
        if self.mean is None or self.stdev is None:
            raise RuntimeError("RevIN denorm requires a preceding norm call")
        channels = features.shape[-1] if output_channels is None else output_channels
        return features * self.stdev[..., :channels] + self.mean[..., :channels]
