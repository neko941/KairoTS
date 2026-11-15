"""Mergeable mean perfect-foresight return (Battery environment).

The best return of each episode with its prices known in advance, whatever the
model predicts: the upper bound that `Return` is read against.
"""

from dataclasses import dataclass

from torch import Tensor

from kairots.environments.Battery import Battery
from kairots.metrics.Return import Return


class ForesightReturn(Return):
    ROLES = ("metric",)
    TASKS = ("Control",)
    NEEDS = ("target",)
    DIRECTION = "max"

    @dataclass(kw_only=True)
    class Config(Return.Config):
        name: str = "ForesightReturn"

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        value = Battery().foresight(price=target).sum()
        self.total = value if self.total is None else self.total + value
        self.count += target.shape[0]
