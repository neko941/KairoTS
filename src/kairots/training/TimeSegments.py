"""Forecasting tasks as consecutive time segments; every task trains on all its windows.

Task 0 is the base segment; later tasks follow in time. `data.split` must be
`TaskWindowRatio` with the same segment lengths, so a window belongs to the task
holding its start and no window crosses a boundary. The scaler fits on task 0's
training rows only and stays fixed. Rounds run per task.
"""

from dataclasses import dataclass
from typing import Any

from kairots.data.splits.base import Segments
from kairots.data.splits.TaskWindowRatio import segment_rows
from kairots.training.base import Continual


class TimeSegments(Continual):
    CONTINUAL = True
    # Reports the task-wise error matrix P with AF, AP and AvgPerf.
    ERROR_MATRIX = True
    TASKS = ("Forecasting",)
    REQUIRES = {
        "task.name": ("Forecasting",),
        "groups": ((),),
        "strategy.name": ("FedAvg",),
        "evaluation.name": ("Frozen",),
        "evaluation.finetune_epochs": (0,),
        "data.split.name": ("TaskWindowRatio",),
    }

    @staticmethod
    def validate(*, experiment: Any) -> None:
        split, training = experiment.data.split, experiment.training
        if (split.base_length, split.task_length, split.task_ranges) != (
            training.base_length,
            training.task_length,
            training.task_ranges,
        ):
            raise ValueError(
                "training and data.split must share segment lengths and task_ranges"
            )
        if training.task_ranges and len(training.task_ranges) != training.periods:
            raise ValueError("task_ranges must contain one range per training period")

    @dataclass(kw_only=True)
    class Config(Continual.Config, Segments.Config):
        name: str = "TimeSegments"

        def __post_init__(self) -> None:
            super().__post_init__()
            Segments.Config.__post_init__(self)

    config: Config

    def windows(
        self,
        *,
        windows: list[tuple[int, int]],
        period: int,
        num_classes: int | None,
    ) -> list[tuple[int, int]]:
        """Windows whose first row lies in task `period`."""
        first, last = segment_rows(
            base_length=self.config.base_length,
            task_length=self.config.task_length,
            period=period,
            task_ranges=self.config.task_ranges,
        )
        return [item for item in windows if first <= item[0] < last]

    def released(
        self,
        *,
        windows: list[tuple[int, int]],
        current_iteration: int,
        iterations: int,
        num_classes: int | None,
    ) -> list[tuple[int, int]]:
        """The whole current task, every round (one local epoch per round)."""
        return self.windows(
            windows=windows,
            period=current_iteration // iterations,
            num_classes=num_classes,
        )
