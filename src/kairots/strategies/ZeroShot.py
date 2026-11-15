"""Score a complete pretrained model without local updates.

An optional source_config names a training experiment JSON. A completed source run
with the same resolved config and prepared-data keys is reused; otherwise it is
trained first. An explicit initialization.run_dir takes precedence. Target scoring
performs no training, and records the selected source run in its resolved config.
"""

from dataclasses import dataclass

from torch import Tensor

from kairots.runtime.contracts import Message
from kairots.strategies.Traditional import Traditional


class ZeroShot(Traditional):
    SOURCE_RUN = True
    REQUIRES = {
        "groups": ((),),
        "client_model": (None,),
        "initialization.name": ("Pretrained",),
        "evaluation.finetune_epochs": (0,),
    }

    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "ZeroShot"
        iterations: int = 1
        source_config: str = ""

        def __post_init__(self) -> None:
            if self.iterations != 1:
                raise ValueError("ZeroShot scores once: iterations must be 1")

    def select_clients(
        self, *, client_ids: list[int], current_iteration: int, seed: int
    ) -> list[int]:
        return []

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        if uplinks:
            raise ValueError("ZeroShot cannot accept training uplinks")
        return global_params
