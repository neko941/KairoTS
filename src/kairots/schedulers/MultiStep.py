"""Decay the learning rate at fractions of the total update count."""

from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR


class MultiStep:
    @dataclass(kw_only=True)
    class Config:
        name: str = "MultiStep"
        fractions: tuple[float, ...] = (0.75, 0.9)
        factor: float = 0.1

        def __post_init__(self) -> None:
            if (
                not all(0 < value < 1 for value in self.fractions)
                or not 0 < self.factor < 1
            ):
                raise ValueError("MultiStep fractions and factor must lie in (0, 1)")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def build(self, *, optimizer: Optimizer, iterations: int) -> LRScheduler:
        return MultiStepLR(
            optimizer,
            milestones=[
                max(1, int(iterations * value)) for value in self.config.fractions
            ],
            gamma=self.config.factor,
        )
