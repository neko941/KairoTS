"""Cross-entropy over class scores (logits)."""

from dataclasses import dataclass

from torch import Tensor
from torch.nn.functional import cross_entropy


class CrossEntropy:
    ROLES = ("loss",)
    TASKS = ("Classification",)
    NEEDS = ("scores", "labels")

    @dataclass(kw_only=True)
    class Config:
        name: str = "CrossEntropy"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def __call__(self, *, prediction: Tensor, target: Tensor) -> Tensor:
        return cross_entropy(prediction, target)
