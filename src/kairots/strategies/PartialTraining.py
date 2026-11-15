"""Partial-training branch of `Heterogeneous`: every
client runs the full model forward but trains only some parameter groups (a group
is one module's own parameters, in `named_parameters` order), uploads only those,
and the server averages each parameter over the clients that trained it, keeping the
global value elsewhere.
Leaves decide which groups each client trains.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

from kairots.runtime.contracts import Job, Message, Result
from kairots.strategies.base import Strategy
from kairots.strategies.Heterogeneous import Heterogeneous


class PartialTraining(Heterogeneous):
    ABSTRACT = True

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        client_id: int
        trainable: Callable[..., list[str]]
        trained: list[str] = field(default_factory=list)

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
            """Freeze every group outside `trainable(client_id, groups)`, then fit.

            A frozen parameter gets no gradient, so the optimizer skips it.
            """
            groups = list(
                dict.fromkeys(
                    name.rpartition(".")[0] for name, _ in model.named_parameters()
                )
            )
            chosen = set(self.trainable(client_id=self.client_id, groups=groups))
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.rpartition(".")[0] in chosen)
                if parameter.requires_grad:
                    self.trained.append(name)
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

    def trainable(self, *, client_id: int, groups: list[str]) -> list[str]:
        raise NotImplementedError

    def client_update(self, *, worker: Any, job: Job) -> Result:
        client = self.Client(client_id=job.client_id, trainable=self.trainable)
        result: Result = worker.update(job=job, client=client)
        uplink = result.uplink
        return replace(
            result,
            uplink=Message(
                sender=uplink.sender,
                receiver=uplink.receiver,
                tensors={
                    name: value
                    for name, value in uplink.tensors.items()
                    if name in client.trained
                },
            ),
        )

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Unweighted mean over the clients that trained each parameter;
        a parameter no client trained keeps its global value."""
        merged = {}
        for name, value in global_params.items():
            updates = [
                message.tensors[name] for message in uplinks if name in message.tensors
            ]
            merged[name] = torch.stack(updates).mean(dim=0) if updates else value
        return merged
