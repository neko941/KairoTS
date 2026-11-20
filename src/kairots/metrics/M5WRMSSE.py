"""Official M5 Accuracy WRMSSE over all 12 hierarchy levels.

`WRMSSE` configured for M5: the 30,490 product-store series summed into the 42,840
official rows, the supplied validation/evaluation weights, and each row's RMSSE scale
from training differences after its first positive sale (the M5 `Estimate WRMSSE.R`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kairots.data.sources.M5._common import (
    SERIES_COUNT,
    aggregation,
    read_sales,
    rmsse_scales,
)
from kairots.metrics.WRMSSE import WRMSSE, HierarchyContext


@dataclass(frozen=True)
class M5Context(HierarchyContext):
    forecast_length: int = 28


class M5WRMSSE(WRMSSE):
    CONTEXT = "wrmsse_context"
    ROLES = ("metric",)
    TASKS = ("Forecasting",)
    NEEDS = ("point_prediction", "target", "wrmsse_context")
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config(WRMSSE.Config):
        name: str = "M5WRMSSE"

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
        hierarchy = aggregation(
            weights_path=root / f"weights_{fold}.csv", identifiers=train.identifiers
        )
        if hierarchy.bottom_count != SERIES_COUNT or len(hierarchy.weights) != 42_840:
            raise ValueError("M5WRMSSE source has an incomplete hierarchy")
        return M5Context(
            row_offsets=hierarchy.row_offsets,
            column_indices=hierarchy.column_indices,
            weights=hierarchy.weights,
            levels=hierarchy.levels,
            bottom_count=hierarchy.bottom_count,
            scales=rmsse_scales(hierarchy=hierarchy, history=train.values.T),
        )
