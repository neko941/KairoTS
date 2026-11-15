"""Centralized baseline: one model trained on all clients' pooled train data.

Nothing crosses the network: the job is addressed from the server to itself.
"""

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from kairots.runtime.contracts import Message
from kairots.strategies.Traditional import Traditional


class Centralized(Traditional):
    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "Centralized"

    def select_clients(
        self, *, client_ids: list[int], current_iteration: int, seed: int
    ) -> list[int]:
        # The selected client's ClientState carries the server optimizer state.
        return [min(client_ids)]

    def build_downlink(
        self, *, global_params: dict[str, Tensor], client_id: int
    ) -> Message:
        return Message(sender="server", receiver="server", tensors=global_params)

    def client_update(self, *, worker: Any, job: Any) -> Any:
        return worker.update(
            job=job, client=self.Client(), client_ids=sorted(worker.records)
        )

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        if len(uplinks) != 1:
            raise ValueError("Centralized requires exactly one result")
        return uplinks[0].tensors
