"""Mergeable top-1 accuracy over class scores."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(kw_only=True)
class Accuracy:
    ROLES = ("metric",)
    TASKS = ("Classification",)
    NEEDS = ("scores", "labels")
    DIRECTION = "max"

    correct: Tensor | None = None
    count: int = 0

    @dataclass(kw_only=True)
    class Config:
        name: str = "Accuracy"

    def __init__(self, *, config: Config | None = None) -> None:
        self.correct = None
        self.count = 0

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.argmax(dim=-1) == target).sum()
        self.correct = value if self.correct is None else self.correct + value
        self.count += target.numel()

    def merge(self, *, other: "Accuracy") -> None:
        if other.correct is not None:
            self.correct = (
                other.correct if self.correct is None else self.correct + other.correct
            )
        self.count += other.count

    def compute(self) -> float:
        if self.correct is None or self.count == 0:
            raise ValueError("Accuracy has no observations")
        return float((self.correct / self.count).item())
