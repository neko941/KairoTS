"""No-federation family: each client trains and keeps its own model."""

from dataclasses import dataclass

from torch import Tensor

from kairots.runtime.contracts import ClientState, Message
from kairots.strategies.base import Strategy


@dataclass(kw_only=True)
class NonFederated(Strategy):
    ABSTRACT = True

    @dataclass(kw_only=True)
    class Config(Strategy.Config):
        name: str = "NonFederated"

    def __init__(self, *, config: Config) -> None:
        self.config = config

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

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        return global_params
