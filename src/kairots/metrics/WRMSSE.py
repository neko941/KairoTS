"""Weighted root mean squared scaled error over a hierarchy (Makridakis et al., 2022).

WRMSSE is the M5 Accuracy competition score ("M5 accuracy competition: Results,
findings, and conclusions", International Journal of Forecasting 38(4), 2022).
Bottom-level forecasts and targets are summed into every series of a hierarchy; each
series' RMSSE divides its mean squared error by a scale fitted on its history; each
level's score is the weighted sum of its series' RMSSE; the score is the mean over
levels. A subclass supplies the hierarchy, weights and scales (`prepare_context`).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import Tensor


@dataclass(frozen=True)
class HierarchyContext:
    """Rows of the summing matrix in CSR form, one weight, level and scale per row."""

    row_offsets: np.ndarray
    column_indices: np.ndarray
    weights: np.ndarray
    levels: np.ndarray
    bottom_count: int
    scales: np.ndarray
    forecast_length: int


class WRMSSE:
    ABSTRACT = True
    CONTEXT = "wrmsse_context"
    ROLES = ("metric",)
    TASKS = ("Forecasting",)
    NEEDS = ("point_prediction", "target", "wrmsse_context")
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config:
        name: str = "WRMSSE"

    def __init__(self, *, config: Config | None = None) -> None:
        self.score: float | None = None

    @classmethod
    def prepare_context(cls, *, experiment: Any) -> HierarchyContext:
        raise NotImplementedError(f"{cls.__name__} supplies the hierarchy and scales")

    @staticmethod
    def aggregate(*, context: HierarchyContext, values: np.ndarray) -> np.ndarray:
        """Sum bottom-level `[time, series]` values into every hierarchy row."""
        if values.ndim != 2 or values.shape[1] != context.bottom_count:
            raise ValueError("values do not match the hierarchy's bottom series")
        result = np.empty((values.shape[0], len(context.weights)), dtype=np.float32)
        for row, (start, end) in enumerate(
            zip(context.row_offsets[:-1], context.row_offsets[1:], strict=True)
        ):
            result[:, row] = values[:, context.column_indices[start:end]].sum(axis=1)
        return result

    def update(
        self, *, prediction: Tensor, target: Tensor, wrmsse_context: HierarchyContext
    ) -> None:
        name = type(self).__name__
        context = wrmsse_context
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError(f"{name} expects matching [1, horizon, series] tensors")
        if prediction.shape[0] != 1 or prediction.shape[1] != context.forecast_length:
            raise ValueError(f"{name} scores exactly one forecast window")
        if prediction.shape[2] != context.bottom_count:
            raise ValueError(f"{name} requires every bottom-level series")
        forecast = self.aggregate(
            context=context, values=prediction[0].detach().to("cpu").numpy()
        )
        actual = self.aggregate(
            context=context, values=target[0].detach().to("cpu").numpy()
        )
        rmsse = np.sqrt(np.mean((forecast - actual) ** 2, axis=0) / context.scales)
        level_scores = [
            float(
                np.dot(
                    context.weights[context.levels == level],
                    rmsse[context.levels == level],
                )
            )
            for level in np.unique(context.levels)
        ]
        self.score = float(np.mean(level_scores))

    def merge(self, *, other: "WRMSSE") -> None:
        if other.score is None:
            return
        if self.score is not None:
            raise ValueError(f"{type(self).__name__} is defined for one window")
        self.score = other.score

    def compute(self) -> float:
        if self.score is None or not np.isfinite(self.score):
            raise ValueError(f"{type(self).__name__} has no valid forecast")
        return self.score
