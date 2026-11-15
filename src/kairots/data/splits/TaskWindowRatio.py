"""Chronological train/test split inside each of consecutive time segments (tasks).

Segment 0 holds the first `base_length` rows; every later segment holds the next
`task_length` rows. Explicit `task_ranges` override these lengths and can leave
unused gaps. Each segment is cut by `ratios` on its own rows, and a window
never crosses a segment boundary, so a window belongs to the segment holding its
start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kairots.data.splits.base import Segments
from kairots.data.splits.CompleteWindowRatio import CompleteWindowRatio


def segment_rows(
    *,
    base_length: int,
    task_length: int,
    period: int,
    task_ranges: tuple[tuple[int, int], ...] = (),
) -> tuple[int, int]:
    """Half-open row range `[first, last)` of task `period`."""
    if task_ranges:
        return task_ranges[period]
    if period == 0:
        return 0, base_length
    first = base_length + (period - 1) * task_length
    return first, first + task_length


class TaskWindowRatio(CompleteWindowRatio):
    @dataclass(kw_only=True)
    class Config(CompleteWindowRatio.Config, Segments.Config):
        name: str = "TaskWindowRatio"

        def __post_init__(self) -> None:
            super().__post_init__()
            Segments.Config.__post_init__(self)
            if self.ratios[1] != 0 or self.cut != "per_client":
                raise ValueError(
                    "TaskWindowRatio has no validation share or shared cut"
                )

    config: Config  # type: ignore[assignment]

    def split(
        self, *, frame: Any, **arguments: Any
    ) -> dict[str, list[tuple[int, int]]]:
        merged: dict[str, list[tuple[int, int]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        period = -1
        while True:
            period += 1
            if self.config.task_ranges and period == len(self.config.task_ranges):
                break
            first, last = segment_rows(
                base_length=self.config.base_length,
                task_length=self.config.task_length,
                period=period,
                task_ranges=self.config.task_ranges,
            )
            if last > len(frame):
                if self.config.task_ranges:
                    raise ValueError("task_ranges exceed the available rows")
                break
            part = super().split(frame=frame.slice(first, last - first), **arguments)
            for name, windows in part.items():
                merged[name] += [
                    (start + first, target + first) for start, target in windows
                ]
        return merged
