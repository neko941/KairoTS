"""Mergeable macro-averaged F1 over class scores.

Per-class true positive, false positive and false negative counts are summed across
batches and clients; F1 is computed once from the pooled counts. A class that never
occurs in the targets or the predictions is left out of the average (as scikit-learn
does with `labels=None`), so a client missing a class is not scored zero for it.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class MacroF1:
    ROLES = ("metric",)
    TASKS = ("Classification",)
    NEEDS = ("scores", "labels")
    DIRECTION = "max"

    # Rows: true positives, false positives, false negatives; one column per class.
    counts: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "MacroF1"

    def __init__(self, *, config: Config | None = None) -> None:
        self.counts = None

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        num_classes = prediction.shape[-1]
        predicted = prediction.argmax(dim=-1)
        hit = predicted == target
        value = torch.stack(
            [
                torch.bincount(target[hit], minlength=num_classes),
                torch.bincount(predicted[~hit], minlength=num_classes),
                torch.bincount(target[~hit], minlength=num_classes),
            ]
        )
        self.counts = value if self.counts is None else self.counts + value

    def merge(self, *, other: "MacroF1") -> None:
        if other.counts is not None:
            self.counts = (
                other.counts if self.counts is None else self.counts + other.counts
            )

    def compute(self) -> float:
        if self.counts is None:
            raise ValueError("MacroF1 has no observations")
        true_positive, false_positive, false_negative = self.counts.double()
        denominator = 2 * true_positive + false_positive + false_negative
        present = denominator > 0
        return float((2 * true_positive[present] / denominator[present]).mean().item())
