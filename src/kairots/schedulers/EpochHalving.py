"""TSLib `lradj="type1"` as its classification experiments call it.

After every `interval`-th completed client update `n` the rate becomes
`learning_rate * 0.5 ** (n - 1)` and stays there until the next multiple, so the
cuts are 2**-4 after update 5, 2**-9 after update 10, and so on. With `interval`
1 it is a halving every update. The rate is read back from the optimizer, so a
client update that resumes from a saved optimizer and scheduler state continues.
"""

from dataclasses import dataclass

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class _EpochHalving(LRScheduler):
    def __init__(  # positional: torch LRScheduler signature.
        self, optimizer: Optimizer, *, interval: int
    ) -> None:
        self.interval = interval
        super().__init__(optimizer)

    def get_lr(self) -> list[float | Tensor]:
        if self.last_epoch == 0 or self.last_epoch % self.interval:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [base * 0.5 ** (self.last_epoch - 1) for base in self.base_lrs]


@dataclass(kw_only=True)
class EpochHalving:
    @dataclass(kw_only=True)
    class Config:
        name: str = "EpochHalving"
        interval: int = 5

        def __post_init__(self) -> None:
            if self.interval < 1:
                raise ValueError("EpochHalving interval must be >= 1")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def build(self, *, optimizer: Optimizer, iterations: int) -> LRScheduler:
        return _EpochHalving(optimizer, interval=self.config.interval)
