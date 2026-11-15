"""FedBABU (Oh et al., ICLR 2022, arXiv:2106.06042v3).

Clients train and share only the body while the head keeps its random start;
each client personalizes by fine-tuning the whole
model before it is scored (`evaluation.finetune_epochs`). Original repository:
https://github.com/jhoon-oh/FedBABU. Full-model fine-tuning is the framework's
evaluation policy; selecting a fine-tuning scope is part of the paper protocol.

The head never changes, so it never travels: the downlink and uplink carry the
body, each client keeps its copy of the starting head, and the server averages the
body. Clients are scored with the server model (`generalization_*`) and after
fine-tuning (`finetuned_*`).
"""

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from kairots.runtime.contracts import ClientState, Message
from kairots.strategies.base import Strategy
from kairots.strategies.Traditional import Traditional

HEAD = "head."


class FedBABU(Traditional):
    FINETUNE = True

    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "FedBABU"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        def fit(
            self,
            *,
            learner: Any,
            model: Any,
            optimizer: Any,
            batches: Any,
            epochs: int,
            device: Any,
            loss_function: Any,
            reference_parameters: dict[str, Tensor],
        ) -> float:
            """Freeze the head, then fit the body."""
            head = [
                parameter
                for name, parameter in model.named_parameters()
                if name.startswith(HEAD)
            ]
            if not head:
                raise ValueError(
                    "FedBABU needs a model whose head has parameters (DLinear's "
                    "forecasting head has none)"
                )
            for parameter in head:
                parameter.requires_grad_(False)
            return super().fit(
                learner=learner,
                model=model,
                optimizer=optimizer,
                batches=batches,
                epochs=epochs,
                device=device,
                loss_function=loss_function,
                reference_parameters=reference_parameters,
            )

    def build_downlink(
        self, *, global_params: dict[str, Tensor], client_id: int
    ) -> Message:
        return Message(
            sender="server",
            receiver=f"client:{client_id}",
            tensors={
                key: value
                for key, value in global_params.items()
                if not key.startswith(HEAD)
            },
        )

    def initial_state(self, *, global_params: dict[str, Tensor]) -> ClientState:
        return ClientState(
            model_state={
                key: value.detach().clone()
                for key, value in global_params.items()
                if key.startswith(HEAD)
            }
        )

    def split_state(
        self, *, parameters: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        uplink = {
            key: value for key, value in parameters.items() if not key.startswith(HEAD)
        }
        kept = {key: value for key, value in parameters.items() if key.startswith(HEAD)}
        return uplink, kept

    def client_parameters(
        self, *, global_params: dict[str, Tensor], state: ClientState
    ) -> dict[str, Tensor]:
        # A client's head equals the server's starting head, so it is scored with
        # the server model.
        return global_params

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Sample-weighted mean of the bodies; the head keeps its start."""
        return {
            **global_params,
            **super().aggregate(uplinks=uplinks, global_params=global_params),
        }
