"""Official M5 Accuracy WRMSSE over all 12 hierarchy levels.

The evaluator sums the 30,490 product-store predictions bottom-up, then applies
the supplied validation/evaluation weights and each aggregate series' RMSSE scale.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from kairots.data.sources.M5._common import (
    SERIES_COUNT,
    Hierarchy,
    aggregate,
    aggregation,
    read_sales,
    rmsse_scales,
)


@dataclass(frozen=True)
class M5Context(Hierarchy):
    scales: np.ndarray
    forecast_length: int = 28


class M5WRMSSE:
    CONTEXT = "m5_context"
    ROLES = ("metric",)
    TASKS = ("Forecasting",)
    NEEDS = ("point_prediction", "target", "m5_context")
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config:
        name: str = "M5WRMSSE"

    def __init__(self, *, config: Config | None = None) -> None:
        self.score: float | None = None

    @classmethod
    def prepare_context(cls, *, experiment: Any) -> M5Context:
        source = experiment.data.source
        fold = {
            "M5Validation": "validation",
            "M5Evaluation": "evaluation",
        }.get(source.name)
        if fold is None:
            raise ValueError("M5WRMSSE requires M5Validation or M5Evaluation")
        root = Path(source.raw_path)
        train = read_sales(path=root / f"sales_train_{fold}.csv")
        weight_file = f"weights_{fold}.csv"
        hierarchy = aggregation(
            weights_path=root / weight_file, identifiers=train.identifiers
        )
        if hierarchy.bottom_count != SERIES_COUNT or len(hierarchy.weights) != 42_840:
            raise ValueError("M5WRMSSE source has an incomplete hierarchy")
        scales = rmsse_scales(hierarchy=hierarchy, history=train.values.T)
        return M5Context(
            row_offsets=hierarchy.row_offsets,
            column_indices=hierarchy.column_indices,
            weights=hierarchy.weights,
            levels=hierarchy.levels,
            bottom_count=hierarchy.bottom_count,
            scales=scales,
        )

    def update(
        self,
        *,
        prediction: Tensor,
        target: Tensor,
        m5_context: M5Context,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("M5WRMSSE expects matching [1, 28, 30,490] tensors")
        if (
            prediction.shape[0] != 1
            or prediction.shape[1] != m5_context.forecast_length
        ):
            raise ValueError("M5WRMSSE scores exactly one 28-day forecast window")
        if prediction.shape[2] != m5_context.bottom_count:
            raise ValueError("M5WRMSSE requires all 30,490 bottom-level series")
        prediction_array = prediction[0].detach().to("cpu").numpy()
        target_array = target[0].detach().to("cpu").numpy()
        forecast = aggregate(hierarchy=m5_context, values=prediction_array)
        actual = aggregate(hierarchy=m5_context, values=target_array)
        rmsse = np.sqrt(np.mean((forecast - actual) ** 2, axis=0) / m5_context.scales)
        level_scores = [
            float(
                np.dot(
                    m5_context.weights[m5_context.levels == level],
                    rmsse[m5_context.levels == level],
                )
            )
            for level in range(1, 13)
        ]
        self.score = float(np.mean(level_scores))

    def merge(self, *, other: M5WRMSSE) -> None:
        if other.score is None:
            return
        if self.score is not None:
            raise ValueError("M5WRMSSE is defined for one full-hierarchy window")
        self.score = other.score

    def compute(self) -> float:
        if self.score is None or not np.isfinite(self.score):
            raise ValueError("M5WRMSSE has no valid forecast")
        return self.score
