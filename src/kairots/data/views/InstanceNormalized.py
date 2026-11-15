"""Panel windows with the TSLib EthanolConcentration instance normalization.

Each sample `[steps, channels]` loses its per-channel mean over steps, then every
step is divided by the population standard deviation across channels at that step
(`1e-5` inside the square root). TSLib's `UEAloader` applies this only to
EthanolConcentration, "for numerical stability", after its dataset standardization.
Nothing is fitted, so train, validation and test samples are treated alike.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.data.views.Offline import Offline


class InstanceNormalized(Offline):
    @dataclass(kw_only=True)
    class Config(Offline.Config):
        name: str = "InstanceNormalized"

    def batches(
        self,
        *,
        values: Tensor,
        windows: list[tuple[int, int]],
        input_length: int,
        output_length: int,
        offset_length: int,
        batch_size: int,
        shuffle: bool,
        generator: torch.Generator,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        if values.dim() != 3:
            raise ValueError("InstanceNormalized needs panel values")
        for batch_x, batch_y in super().batches(
            values=values,
            windows=windows,
            input_length=input_length,
            output_length=output_length,
            offset_length=offset_length,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        ):
            centered = batch_x - batch_x.mean(dim=1, keepdim=True)
            spread = torch.sqrt(
                centered.var(dim=2, keepdim=True, unbiased=False) + 1e-5
            )
            yield centered / spread, batch_y
