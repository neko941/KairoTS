"""Adam optimization with configurable weight decay (Kingma and Ba, ICLR 2015)."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.optim import Adam as TorchAdam
from torch.optim import Optimizer

from kairots.optimizers.base import Optimizer as OptimizerSettings


class Adam:
    @dataclass(kw_only=True)
    class Config(OptimizerSettings.Config):
        name: str = "Adam"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def build(
        self, *, parameters: Iterable[Tensor], learning_rate: float, fused: bool
    ) -> Optimizer:
        options: dict[str, Any] = {
            "lr": learning_rate,
            "weight_decay": self.config.weight_decay,
        }
        if fused and "fused" in TorchAdam.__init__.__code__.co_varnames:
            options["fused"] = True
        return TorchAdam(parameters, **options)
