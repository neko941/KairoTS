"""Imputation whose model also sees the observation mask and calendar marks.

Same hiding as `Imputation`; the model input is `[values | observed mask | calendar]`
(values zeroed where hidden), so a model can normalize over observed positions only
and embed time marks, as TSLib's `imputation` does. The target is the true value at
hidden positions and NaN elsewhere.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.tasks.base import Timed
from kairots.tasks.Imputation import Imputation


class MaskedImputation(Imputation):
    HEAD = "Imputation"

    @dataclass(kw_only=True)
    class Config(Imputation.Config, Timed.Config):
        name: str = "MaskedImputation"
        input_channels: tuple[str, ...] = ()

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
        channels = len(self.config.input_channels) or batch_y.shape[-1]
        calendar = batch_y[..., channels:]
        values, target = super().prepare(
            batch_x=batch_x,
            batch_y=batch_y[..., :channels],
            output_length=output_length,
            generator=generator,
        )
        # Observed: not hidden (target is NaN) and natively present.
        observed = (~target.isfinite() & batch_y[..., :channels].isfinite()).to(
            dtype=values.dtype
        )
        return torch.cat([values, observed, calendar], dim=-1), target
