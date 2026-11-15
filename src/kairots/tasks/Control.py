"""Sequential decision task on series windows (`environments/Battery.py`).

The policy reads a window's history and chooses one action per step of the
`output_length` horizon. The horizon's prices (the last channel) are the
environment's hidden state: they become the target, never a model input. Runs
report the greedy policy's `Return` and the perfect-foresight `ForesightReturn`.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class Control:
    HEAD = "Control"
    RECONSTRUCTS = False
    LOSS = "Return"
    METRICS = ("Return", "ForesightReturn")

    @dataclass(kw_only=True)
    class Config:
        name: str = "Control"

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
        return batch_x, batch_y[..., -1]
