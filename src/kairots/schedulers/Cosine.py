"""Cosine annealing (torch `CosineAnnealingLR`): the rate falls from
`learning_rate` to `minimum_rate`. With `span` "run" it falls across
`strategy.iterations`, one step per client update; with "round" it restarts every
client update and falls across that update's optimizer steps."""

from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


@dataclass(kw_only=True)
class Cosine:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Cosine"
        minimum_rate: float = 0.0
        span: str = "run"

        def __post_init__(self) -> None:
            if self.span not in ("run", "round"):
                raise ValueError("Cosine span is 'run' or 'round'")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def build(self, *, optimizer: Optimizer, iterations: int) -> LRScheduler:
        return CosineAnnealingLR(
            optimizer, T_max=iterations, eta_min=self.config.minimum_rate
        )
