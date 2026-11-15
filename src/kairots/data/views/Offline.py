"""Index and gather fixed windows from one client tensor."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class Offline:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Offline"
        # Shuffled (training) batches drop the final short batch, as the benchmark
        # data loaders do; evaluation batches never drop windows.
        drop_last: bool = False

    def __init__(self, *, config: Config) -> None:
        self.config = config

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
        order = (
            torch.randperm(len(windows), generator=generator)
            if shuffle
            else torch.arange(len(windows))
        )
        # Panel data: values is [samples, steps, channels] and a window is
        # (sample, label). Series: a window is (start, target) row offsets.
        first, second = torch.tensor(windows, dtype=torch.long).reshape(-1, 2).T
        steps = (torch.arange(input_length), torch.arange(output_length))
        count = len(windows)
        if shuffle and self.config.drop_last:
            count -= count % batch_size
        for start in range(0, count, batch_size):
            index = order[start : start + batch_size]
            if values.dim() == 3:
                batch_x = values[first[index]]
                batch_y = batch_x if output_length else second[index]
            else:
                batch_x = values[first[index, None] + steps[0]]
                batch_y = values[second[index, None] + steps[1]]
            yield batch_x, batch_y
