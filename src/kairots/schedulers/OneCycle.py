"""One-cycle learning-rate schedule advanced after every optimizer step."""

from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR


@dataclass(kw_only=True)
class OneCycle:
    STEP_PER_BATCH = True

    @dataclass(kw_only=True)
    class Config:
        name: str = "OneCycle"
        maximum_rate: float = 0.01

        def __post_init__(self) -> None:
            if self.maximum_rate <= 0:
                raise ValueError("OneCycle maximum_rate must be positive")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def build(self, *, optimizer: Optimizer, iterations: int) -> LRScheduler:
        if iterations < 1:
            raise ValueError("OneCycle needs at least one optimizer step")
        return OneCycleLR(
            optimizer, max_lr=self.config.maximum_rate, total_steps=iterations
        )
