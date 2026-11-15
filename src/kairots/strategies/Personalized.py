"""Personalized family: a federated global model plus a personal model
per client, kept in `ClientState.extra`; clients are scored with their personal model.
"""

from dataclasses import dataclass

from torch import Tensor

from kairots.runtime.contracts import ClientState
from kairots.strategies.Traditional import Traditional


class Personalized(Traditional):
    ABSTRACT = True

    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "Personalized"

    def initial_state(self, *, global_params: dict[str, Tensor]) -> ClientState:
        return ClientState(
            extra={
                "personal": {
                    key: value.detach().clone() for key, value in global_params.items()
                }
            }
        )

    def client_parameters(
        self, *, global_params: dict[str, Tensor], state: ClientState
    ) -> dict[str, Tensor]:
        return state.extra.get("personal", global_params)
