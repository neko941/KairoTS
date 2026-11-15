"""No scaling: mean 0 and scale 1 on every channel, so values pass through unchanged."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class Identity:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Identity"

    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, *, values: Tensor, config: Config) -> Identity:
        mean = values.mean(dim=0)
        return cls(mean=torch.zeros_like(mean), scale=torch.ones_like(mean))
