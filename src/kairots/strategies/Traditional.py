"""Traditional centralized-server select, broadcast and aggregate flow."""

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.runtime.contracts import Message
from kairots.strategies.base import Strategy


@dataclass(kw_only=True)
class Traditional(Strategy):
    ABSTRACT = True

    @dataclass(kw_only=True)
    class Config(Strategy.Config):
        name: str = "Traditional"
        join_ratio: float = 1.0

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def select_clients(
        self, *, client_ids: list[int], current_iteration: int, seed: int
    ) -> list[int]:
        count = max(1, math.ceil(len(client_ids) * self.config.join_ratio))
        return sorted(
            random.Random(seed).sample(client_ids, min(count, len(client_ids)))
        )

    def build_downlink(
        self, *, global_params: dict[str, Tensor], client_id: int
    ) -> Message:
        return Message(
            sender="server", receiver=f"client:{client_id}", tensors=global_params
        )

    def uplink_scalars(
        self, *, tensors: dict[str, Tensor], train_samples: int
    ) -> dict[str, float]:
        """The sample count that weights the server's average; nothing to weight,
        nothing sent."""
        return {"train_samples": float(train_samples)} if tensors else {}

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Sample-weighted mean of client parameters, weighted by the sent counts."""
        if not uplinks:
            raise ValueError("aggregation requires nonempty client uplinks")
        if not any(message.tensors for message in uplinks):
            return global_params
        counts = [message.scalars["train_samples"] for message in uplinks]
        if any(count <= 0 for count in counts):
            raise ValueError(
                "aggregation requires nonempty client uplinks with positive sample "
                "counts"
            )
        total = sum(counts)
        keys = tuple(uplinks[0].tensors)
        if any(tuple(message.tensors) != keys for message in uplinks):
            raise ValueError("client parameter keys differ")
        return {
            key: sum(
                (
                    message.tensors[key] * (count / total)
                    for count, message in zip(counts, uplinks, strict=True)
                ),
                torch.zeros_like(uplinks[0].tensors[key]),
            )
            for key in keys
        }
