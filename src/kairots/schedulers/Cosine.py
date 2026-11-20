"""Cosine annealing (torch `CosineAnnealingLR`): the rate falls from
`learning_rate` to `minimum_rate`. With `span` "run" it falls across
`strategy.iterations`, one step per client update; with "round" it restarts every
client update and falls across that update's optimizer steps."""

import math
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler


@dataclass(kw_only=True)
class Cosine:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Cosine"
        minimum_rate: float = 0.0
        span: str = "run"
        # Fall across this many iterations, then hold `minimum_rate` (0: the whole run).
        decay_iterations: int = 0

        def __post_init__(self) -> None:
            if self.span not in ("run", "round"):
                raise ValueError("Cosine span is 'run' or 'round'")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def build(self, *, optimizer: Optimizer, iterations: int) -> LRScheduler:
        decay = self.config.decay_iterations
        if decay:
            # Rebuilt every client update on an optimizer whose rate is already
            # annealed (0 once the decay ends): anneal from the starting rate.
            group = optimizer.param_groups[0]
            base = group.get("initial_lr", group["lr"])
            floor = self.config.minimum_rate
            return LambdaLR(
                optimizer,
                lambda step: (
                    (
                        floor
                        + (base - floor)
                        * (1 + math.cos(math.pi * min(step, decay) / decay))
                        / 2
                    )
                    / base
                ),
            )
        return CosineAnnealingLR(
            optimizer, T_max=iterations, eta_min=self.config.minimum_rate
        )
