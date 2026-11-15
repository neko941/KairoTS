"""FedPAQ, Reisizadeh et al. (AISTATS 2020), arXiv:1909.13014v4: norm-based
stochastic quantization of the whole update to `levels` levels per sign; the wire
carries one sign bit and `ceil(log2(levels + 1))` magnitude bits (magnitudes 0 to
`levels`) per value plus a 32-bit norm. `levels = 0` sends the update unquantized."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.strategies.Quantized import Quantized


class FedPAQ(Quantized):
    @dataclass(kw_only=True)
    class Config(Quantized.Config):
        name: str = "FedPAQ"
        levels: int = 8

        def __post_init__(self) -> None:
            if self.levels < 0:
                raise ValueError("FedPAQ levels must be >= 0")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def quantize(self, *, vector: Tensor, generator: torch.Generator) -> Tensor:
        levels = self.config.levels
        norm = vector.norm()
        if levels == 0 or norm == 0:
            return vector
        scaled = vector.abs() / norm * levels
        lower = scaled.floor()
        noise = torch.rand(vector.shape, generator=generator).to(vector.device)
        level = lower + (noise < scaled - lower)
        return norm * vector.sign() * level / levels

    def encoded_bytes(self, *, numel: int) -> int | None:
        if self.config.levels == 0:
            return None
        # Magnitudes 0..levels: levels + 1 values.
        magnitude_bits = math.ceil(math.log2(self.config.levels + 1))
        return math.ceil((numel * (magnitude_bits + 1) + 32) / 8)
