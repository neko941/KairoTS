"""Stochastic gradient descent with optional momentum and weight decay."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch import Tensor, optim
from torch.optim import Optimizer

from kairots.optimizers.base import Optimizer as OptimizerSettings


class SGD:
    @dataclass(kw_only=True)
    class Config(OptimizerSettings.Config):
        name: str = "SGD"
        momentum: float = 0.0

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def build(
        self, *, parameters: Iterable[Tensor], learning_rate: float, fused: bool
    ) -> Optimizer:
        options: dict[str, Any] = {
            "lr": learning_rate,
            "momentum": self.config.momentum,
            "weight_decay": self.config.weight_decay,
        }
        if fused and "fused" in optim.SGD.__init__.__code__.co_varnames:
            options["fused"] = True
        return optim.SGD(parameters, **options)
