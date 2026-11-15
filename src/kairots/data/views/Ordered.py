"""Time-ordered stream: batches follow window order and are never shuffled, as an
online or prequential regime reads them."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.data.views.Offline import Offline


class Ordered(Offline):
    @dataclass(kw_only=True)
    class Config(Offline.Config):
        name: str = "Ordered"

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
        return super().batches(
            values=values,
            windows=windows,
            input_length=input_length,
            output_length=output_length,
            offset_length=offset_length,
            batch_size=batch_size,
            shuffle=False,
            generator=generator,
        )
