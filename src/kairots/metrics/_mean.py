"""Tensor sums and counts shared by mergeable mean scores."""

from typing import Any

from torch import Tensor


def add(*, metric: Any, total: Tensor, count: Tensor | int) -> None:
    value = total.detach()
    metric.total = value if metric.total is None else metric.total + value
    metric.count = count if metric.count is None else metric.count + count


def merge(*, metric: Any, other: Any) -> None:
    if other.total is not None:
        add(metric=metric, total=other.total, count=other.count)


def mean(*, metric: Any) -> float:
    if metric.total is None or metric.count is None or not metric.count:
        raise ValueError(f"{type(metric).__name__} has no observations")
    return float((metric.total / metric.count).item())
