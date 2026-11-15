"""Imputation inputs carry values and an explicit conditional observation mask."""

from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.tasks.base import Timed
from kairots.tasks.Imputation import Imputation


class ProbabilisticImputation(Imputation):
    HEAD = "Samples"
    LOSS = "SampleCRPS"
    METRICS = ("NormalizedCRPS", "MedianMAE")

    @dataclass(kw_only=True)
    class Config(Imputation.Config, Timed.Config):
        name: str = "ProbabilisticImputation"
        input_channels: tuple[str, ...] = ()
        target_pattern: tuple[tuple[bool, ...], ...] = ()

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def prepare(
        self,
        *,
        batch_x: Tensor,
        batch_y: Tensor,
        output_length: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if generator is None:
            raise ValueError("Imputation needs a mask generator")
        channels = len(self.config.input_channels) or batch_y.shape[-1]
        time = batch_y[..., channels:]
        batch_y = batch_y[..., :channels]
        observed = batch_y.isfinite()
        hidden = torch.zeros(observed.shape, device=observed.device, dtype=torch.bool)
        for index in range(len(batch_y)):
            positions = observed[index].flatten().nonzero().flatten()
            count = int(len(positions) * self.config.mask_ratio)
            choices = torch.randperm(len(positions), generator=generator)[:count]
            hidden[index].view(-1)[positions[choices]] = True
        if self.config.target_pattern:
            pattern = torch.tensor(self.config.target_pattern, dtype=torch.bool)
            if pattern.shape != batch_y.shape[1:]:
                raise ValueError("target_pattern must match the target window")
            hidden = observed & ~pattern
        if not hidden.any():
            raise ValueError("imputation mask has no observed prediction targets")
        conditional = observed & ~hidden
        values = torch.where(conditional, batch_y, 0.0)
        return torch.cat(
            [values, conditional.to(dtype=values.dtype), time], dim=-1
        ), torch.where(hidden, batch_y, torch.nan)
