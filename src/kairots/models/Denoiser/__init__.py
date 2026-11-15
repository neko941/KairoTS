"""MLP noise predictor for `Generation` (DDPM): the flattened noised window plus a
sinusoidal embedding of the diffusion step (the input's last channel), through
residual SiLU blocks, to the predicted noise of every window value. It mixes every
step and channel of the window."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kairots.models.base import FeatureWidth


class Denoiser(nn.Module):
    TASKS = ("Generation",)

    @dataclass(kw_only=True)
    class Config(FeatureWidth.Config):
        name: str = "Denoiser"
        input_length: int = 24
        input_channels: int = 7
        width: int = 256
        depth: int = 2

        def __post_init__(self) -> None:
            if self.width < 2 or self.width % 2:
                raise ValueError("Denoiser width must be even")

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        size = config.input_length * config.input_channels
        self.width = config.width
        self.embed = nn.Sequential(
            nn.Linear(config.width, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.encode = nn.Linear(size, config.width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(config.width, config.width))
            for _ in range(config.depth)
        )
        self.decode = nn.Sequential(nn.SiLU(), nn.Linear(config.width, size))

    def forward(self, *, batch_x: Tensor) -> Tensor:
        window = batch_x[..., :-1]
        steps = batch_x[:, 0, -1]
        half = self.width // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=batch_x.device, dtype=batch_x.dtype)
            / half
        )
        angles = steps[:, None] * frequencies
        hidden = self.encode(window.flatten(1)) + self.embed(
            torch.cat([angles.sin(), angles.cos()], dim=-1)
        )
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.decode(hidden).view(window.shape)
