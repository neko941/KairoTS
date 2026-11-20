"""Round flow and client update contracts."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.runtime.contracts import ClientState, Message

# Members a family may own; two parent families owning one is a clash.
HOOKS = (
    "Config",
    "Client",
    "setup",
    "select_clients",
    "build_downlink",
    "initial_state",
    "client_parameters",
    "personal_model",
    "split_state",
    "uplink_scalars",
    "exchange",
    "client_update",
    "aggregate",
    "broadcast",
)


@dataclass(kw_only=True)
class Strategy:
    ABSTRACT = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject a class whose parent families both override one hook, unless the
        class defines that hook itself (never resolve a clash silently by MRO)."""
        super().__init_subclass__(**kwargs)
        for hook in HOOKS:
            if hook in cls.__dict__:
                continue
            owners = {
                next(owner for owner in base.__mro__ if hook in owner.__dict__)
                for base in cls.__bases__
                if issubclass(base, Strategy)
            }
            if any(
                not issubclass(first, second) and not issubclass(second, first)
                for first in owners
                for second in owners
            ):
                names = ", ".join(sorted(owner.__name__ for owner in owners))
                raise TypeError(
                    f"{cls.__name__}: parent families {names} both define `{hook}`; "
                    f"define `{hook}` on {cls.__name__}"
                )

    @dataclass(kw_only=True)
    class Config:
        name: str = ""
        iterations: int = 3

    @dataclass(kw_only=True)
    class Client:
        def proximal_term(
            self, *, model: nn.Module, reference_parameters: dict[str, Tensor]
        ) -> Tensor:
            return torch.zeros((), device=next(model.parameters()).device)

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
            loss = 0.0
            for _ in range(epochs):
                loss = learner.fit_epoch(
                    model=model,
                    optimizer=optimizer,
                    batches=batches,
                    device=device,
                    loss_function=loss_function,
                    regularizer=lambda: self.proximal_term(
                        model=model, reference_parameters=reference_parameters
                    ),
                )
            return loss

    def setup(self, *, experiment: Any) -> None:
        """Server-side preparation before round 0, from the resolved experiment
        (`FedTREND` keeps the model config and lengths for its synthetic data).
        Default: nothing."""

    def select_clients(
        self, *, client_ids: list[int], current_iteration: int, seed: int
    ) -> list[int]:
        raise NotImplementedError

    def build_downlink(self, *, global_params: dict[str, Any], client_id: int) -> Any:
        raise NotImplementedError

    def initial_state(self, *, global_params: dict[str, Tensor]) -> ClientState:
        """Each client's state before the first round (empty by default)."""
        return ClientState()

    def client_parameters(
        self, *, global_params: dict[str, Tensor], state: ClientState
    ) -> dict[str, Tensor]:
        """The parameters a client is evaluated with: the global model overlaid by
        what the client keeps."""
        return {**global_params, **state.model_state}

    def personal_model(self, *, model: nn.Module, experiment: Any) -> nn.Module:
        """The module a client is scored with as personalization, built once from
        the client model; `client_parameters` fill it. Default: the model itself."""
        return model

    def split_state(
        self, *, parameters: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Split trained parameters into (uplink, kept by the client)."""
        return parameters, {}

    def uplink_scalars(
        self, *, tensors: dict[str, Tensor], train_samples: int
    ) -> dict[str, float]:
        """Scalars a client sends beside its uplink tensors, billed 4 bytes each.

        The server reads only what the uplink carries, so a method sends here what
        its paper sends (a sample count, a step count, a score). Default: none.
        """
        return {}

    def exchange(self, *, results: list[Any]) -> list[Message]:
        """Peer-to-peer step after local training (decentralized methods).

        May replace each result's `state` (for example with mixed parameters) and
        returns the messages peers sent, which are billed. Default: none.
        """
        return []

    def client_update(self, *, worker: Any, job: Any) -> Any:
        return worker.update(job=job, client=self.Client())

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Any]:
        raise NotImplementedError

    def broadcast(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> list[Message]:
        """Messages sent after aggregation, such as a one-shot fitted model."""
        return []
