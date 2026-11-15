"""Reservoir replay buffer (Vitter's Algorithm R), as in experience replay for
continual learning (Chaudhry et al., 2019, "On Tiny Episodic Memories").

The buffer holds at most `capacity` windows. Every admitted window has the same
chance, capacity / seen, of being in the buffer, whatever its position in the
stream. A seeded generator decides admission and sampling, and `reset` restarts
it, so one client's stream always replays the same windows.
"""

from dataclasses import dataclass
from typing import Any

import torch

from kairots.replay.base import Buffer


@dataclass(kw_only=True)
class Reservoir:
    @dataclass(kw_only=True)
    class Config(Buffer.Config):
        name: str = "Reservoir"
        seed: int = 0

        def __post_init__(self) -> None:
            if self.capacity < 0:
                raise ValueError("capacity must be non-negative")

    def __init__(self, *, config: Config) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Empty the buffer and restart its generator."""
        self.items: list[Any] = []
        self.seen = 0
        self.generator = torch.Generator().manual_seed(self.config.seed)

    def admit(self, *, items: list[Any]) -> None:
        """Offer each item once, in stream order."""
        for item in items:
            self.seen += 1
            if len(self.items) < self.config.capacity:
                slot = len(self.items)
            else:
                slot = int(torch.randint(self.seen, (1,), generator=self.generator))
                if slot >= self.config.capacity:
                    continue
            # Retained windows own storage instead of keeping their batch alive.
            retained = (
                tuple(
                    value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for value in item
                )
                if isinstance(item, tuple)
                else item
            )
            if slot == len(self.items):
                self.items.append(retained)
            else:
                self.items[slot] = retained

    def sample(self, *, count: int) -> list[Any]:
        """Up to `count` distinct buffered items, drawn uniformly."""
        if count >= len(self.items):
            return list(self.items)
        order = torch.randperm(len(self.items), generator=self.generator)
        return [self.items[index] for index in order[:count].tolist()]
