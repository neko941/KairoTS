"""Decentralized family: every client keeps its own model,
trains locally, then averages with its topology neighbours (one gossip step,
Metropolis weights); there is no server model."""

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from kairots.registry import get
from kairots.runtime.contracts import ClientState, Message
from kairots.strategies.base import Strategy


@dataclass(kw_only=True)
class Decentralized(Strategy):
    ABSTRACT = True

    @dataclass(kw_only=True)
    class Config(Strategy.Config):
        name: str = "Decentralized"
        topology: str = "FullyConnected"

    def __init__(self, *, config: Config) -> None:
        self.config = config
        topology_type = get(category="topologies", name=config.topology)
        self.topology = topology_type(config=topology_type.Config())

    def select_clients(
        self, *, client_ids: list[int], current_iteration: int, seed: int
    ) -> list[int]:
        return sorted(client_ids)

    def build_downlink(
        self, *, global_params: dict[str, Tensor], client_id: int
    ) -> Message:
        return Message(sender="server", receiver=f"client:{client_id}")

    def initial_state(self, *, global_params: dict[str, Tensor]) -> ClientState:
        return ClientState(
            model_state={
                key: value.detach().clone() for key, value in global_params.items()
            }
        )

    def split_state(
        self, *, parameters: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return {}, parameters

    def exchange(self, *, results: list[Any]) -> list[Message]:
        client_ids = sorted(result.operation_id[1] for result in results)
        if not client_ids:
            return []
        # Build the topology over the selected clients only.
        position_neighbors = self.topology.neighbors(num_nodes=len(client_ids))
        neighbors = {
            client_id: [client_ids[position] for position in position_neighbors[index]]
            for index, client_id in enumerate(client_ids)
        }
        result_by_client = {result.operation_id[1]: result for result in results}
        snapshot = {
            client_id: {
                key: value.detach().clone()
                for key, value in result_by_client[client_id].state.model_state.items()
            }
            for client_id in client_ids
        }
        degree = {client_id: len(neighbors[client_id]) for client_id in client_ids}
        messages: list[Message] = []
        for client_id in client_ids:
            peer_ids = [client_id, *neighbors[client_id]]
            peer_weights = [
                1.0 / (1 + max(degree[client_id], degree[peer_id]))
                for peer_id in peer_ids[1:]
            ]
            weights = [1.0 - sum(peer_weights), *peer_weights]
            mixed = {
                key: sum(
                    (
                        snapshot[peer_id][key] * weight
                        for peer_id, weight in zip(peer_ids, weights)
                    ),
                    torch.zeros_like(snapshot[client_id][key]),
                )
                for key in snapshot[client_id]
            }
            result_by_client[client_id].state = replace(
                result_by_client[client_id].state, model_state=mixed
            )
            messages.extend(
                Message(
                    sender=f"client:{client_id}",
                    receiver=f"client:{peer_id}",
                    tensors=snapshot[client_id],
                )
                for peer_id in neighbors[client_id]
            )
        return messages

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        return global_params
