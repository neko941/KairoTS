"""Mergeable mean episode return of the greedy actions (Battery environment)."""

from dataclasses import dataclass

from torch import Tensor

from kairots.environments.Battery import Battery


@dataclass(kw_only=True)
class Return:
    ROLES = ("loss", "metric")
    TASKS = ("Control",)
    NEEDS = ("scores", "target")
    DIRECTION = "max"

    total: Tensor | None = None
    count: int = 0

    @dataclass(kw_only=True)
    class Config:
        name: str = "Return"

    def __init__(self, *, config: Config | None = None) -> None:
        self.total = None
        self.count = 0
        self.environment = Battery()

    def _returns(self, *, prediction: Tensor, target: Tensor) -> Tensor:
        _, rewards, _ = self.environment.rollout(
            logits=prediction,
            price=target,
            choose=lambda logits: logits.argmax(dim=-1),
        )
        return rewards.sum(dim=1)

    def __call__(self, *, prediction: Tensor, target: Tensor) -> Tensor:
        return -self._returns(prediction=prediction, target=target).mean()

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        value = self._returns(prediction=prediction, target=target).sum()
        self.total = value if self.total is None else self.total + value
        self.count += target.shape[0]

    def merge(self, *, other: "Return") -> None:
        if other.total is not None:
            self.total = other.total if self.total is None else self.total + other.total
        self.count += other.count

    def compute(self) -> float:
        if self.total is None or not self.count:
            raise ValueError("Return has no episodes")
        return float((self.total / self.count).item())
