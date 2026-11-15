"""Narrow settings shared by update loops and gradient optimization clients."""

from dataclasses import dataclass, field
from typing import Any

from kairots.optimizers.Adam import Adam


class Learner:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Learner"
        epochs: int = 1
        batches_per_epoch: int | None = None

        def __post_init__(self) -> None:
            if self.batches_per_epoch is not None and self.batches_per_epoch < 1:
                raise ValueError("learner.batches_per_epoch must be positive")


class Learning:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Learning"
        learning_rate: float = 0.0001
        optimizer: Any = field(default_factory=Adam.Config)
