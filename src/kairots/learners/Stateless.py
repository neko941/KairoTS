"""No gradient fitting: no optimizer and zero epochs.

Used by parameter-free models (Naive) and strategies that fit sufficient statistics
directly (FedRidge).
"""

from dataclasses import dataclass

from torch import nn

from kairots.learners.base import Learner


@dataclass(kw_only=True)
class Stateless:
    @dataclass(kw_only=True)
    class Config(Learner.Config):
        name: str = "Stateless"
        epochs: int = 0

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def optimizer(self, *, model: nn.Module) -> None:
        return None
