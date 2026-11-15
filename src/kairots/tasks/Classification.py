"""Single-label sequence classification task semantics."""

from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class Classification:
    """Whole sequences in, one class index out; targets are integer labels."""

    HEAD = "Classification"
    RECONSTRUCTS = False
    LOSS = "CrossEntropy"
    METRICS = ("Accuracy", "MacroF1")

    @dataclass(kw_only=True)
    class Config:
        name: str = "Classification"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def prepare(
        self,
        *,
        batch_x: Any,
        batch_y: Any,
        output_length: int,
        generator: Any = None,
    ) -> tuple[Any, Any]:
        return batch_x, batch_y
