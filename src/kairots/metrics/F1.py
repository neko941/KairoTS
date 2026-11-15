"""Mergeable point-wise F1 of binary anomaly predictions against per-step labels.

No point adjustment; see `AdjustedF1` for TSLib's headline number. F1 is undefined
(NaN) when there are no positive labels and no positive predictions.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class F1:
    ROLES = ("metric",)
    TASKS = ("AnomalyDetection",)
    NEEDS = ("anomaly_predictions", "labels")
    DIRECTION = "max"
    PHASES = ("test",)

    # True positives, false positives, false negatives.
    counts: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "F1"

    def __init__(self, *, config: Config | None = None) -> None:
        self.counts = None

    @staticmethod
    def adjust(*, prediction: Tensor, target: Tensor) -> Tensor:
        """Return `prediction` unchanged; `AdjustedF1` overrides this."""
        return prediction

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        """Add one client's whole test timeline (bool predictions, 0/1 labels)."""
        target = target.bool()
        prediction = self.adjust(prediction=prediction.bool(), target=target)
        value = torch.stack(
            [
                (prediction & target).sum(),
                (prediction & ~target).sum(),
                (~prediction & target).sum(),
            ]
        )
        self.counts = value if self.counts is None else self.counts + value

    def merge(self, *, other: "F1") -> None:
        if other.counts is not None:
            self.counts = (
                other.counts if self.counts is None else self.counts + other.counts
            )

    def compute(self) -> float:
        if self.counts is None:
            raise ValueError(f"{type(self).__name__} has no observations")
        true_positive, false_positive, false_negative = self.counts.double()
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator == 0:
            return float("nan")
        return float((2 * true_positive / denominator).item())
