"""Jitter (TSLib `utils/augmentation.py`, Um et al. 2017): add Gaussian noise with
standard deviation `sigma` to training inputs; targets are untouched."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class Jitter:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Jitter"
        sigma: float = 0.03

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def apply(self, *, batch_x: Tensor, generator: torch.Generator) -> Tensor:
        noise = torch.randn(batch_x.shape, generator=generator, dtype=batch_x.dtype)
        return batch_x + self.config.sigma * noise.to(batch_x.device)
