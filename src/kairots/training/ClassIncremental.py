"""Class-incremental tasks with disjoint examples released each communication round.

The scaler fits on the first round's released examples only and stays fixed.
"""

from dataclasses import dataclass
from typing import Any

from kairots.training.base import Continual


class ClassIncremental(Continual):
    CONTINUAL = True
    TASKS = ("Classification",)
    REQUIRES = {
        "task.name": ("Classification",),
        "groups": ((),),
        "strategy.name": ("FedAvg", "FLwF2"),
        "evaluation.name": ("Frozen",),
        "evaluation.finetune_epochs": (0,),
    }

    @staticmethod
    def validate(*, experiment: Any) -> None:
        num_classes = getattr(experiment.data.source, "num_classes", None)
        if num_classes is None:
            raise ValueError("ClassIncremental needs data.source.num_classes")
        if not 2 <= experiment.training.periods <= num_classes:
            raise ValueError("Continual periods must not exceed num_classes")

    @dataclass(kw_only=True)
    class Config(Continual.Config):
        name: str = "ClassIncremental"

    def windows(
        self, *, windows: list[tuple[int, int]], period: int, num_classes: int | None
    ) -> list[tuple[int, int]]:
        """Consecutive class groups; each class belongs to exactly one task."""
        if num_classes is None:
            raise ValueError("ClassIncremental needs labeled classification windows")
        return [
            item
            for item in windows
            if item[1] * self.config.periods // num_classes == period
        ]
