"""Binary area under the ROC curve over class scores (ties share their mean rank).

Exact AUROC is not a sum of per-batch statistics, so the accumulator keeps every
score and label and merges by concatenation.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class AUROC:
    ROLES = ("metric",)
    TASKS = ("Classification",)
    NEEDS = ("scores", "labels")
    DIRECTION = "max"

    scores: list[Tensor]
    labels: list[Tensor]

    @dataclass(kw_only=True)
    class Config:
        name: str = "AUROC"

    def __init__(self, *, config: Config | None = None) -> None:
        self.scores = []
        self.labels = []

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape[-1] != 2:
            raise ValueError("AUROC needs two class scores")
        # Logit margin: monotone in the positive-class probability.
        self.scores.append((prediction[..., 1] - prediction[..., 0]).detach().double())
        self.labels.append(target.detach())

    def merge(self, *, other: "AUROC") -> None:
        self.scores += other.scores
        self.labels += other.labels

    def compute(self) -> float:
        if not self.scores:
            raise ValueError("AUROC has no observations")
        scores, labels = torch.cat(self.scores), torch.cat(self.labels) == 1
        positives = int(labels.sum())
        negatives = labels.numel() - positives
        if positives == 0 or negatives == 0:
            return float("nan")
        _, inverse, counts = torch.unique(
            scores, return_inverse=True, return_counts=True
        )
        mean_rank = counts.cumsum(0).double() - (counts.double() - 1) / 2
        rank_sum = mean_rank[inverse][labels].sum()
        return float(
            ((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))
        )
