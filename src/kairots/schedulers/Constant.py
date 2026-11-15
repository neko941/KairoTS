"""Constant learning rate: every round trains at the learner's `learning_rate`."""

from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LRScheduler


@dataclass(kw_only=True)
class Constant:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Constant"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def build(self, *, optimizer: Optimizer, iterations: int) -> LRScheduler:
        return ConstantLR(optimizer, factor=1.0, total_iters=0)
