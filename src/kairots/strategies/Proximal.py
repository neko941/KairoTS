"""Proximal family (the FedProx local objective): each client adds
`mu / 2 * ||w - w_start||^2` to its loss, where `w_start` is the model it started the
round from (the global model, or its own mixed model when decentralized). It owns only
the local objective, so a method combines it with a family that decides who talks.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.runtime.contracts import Job, Result
from kairots.strategies.base import Strategy


class Proximal(Strategy):
    ABSTRACT = True
    config: "Proximal.Config"

    @dataclass(kw_only=True)
    class Config(Strategy.Config):
        name: str = "Proximal"
        mu: float = 0.01

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        mu: float

        def proximal_term(
            self, *, model: nn.Module, reference_parameters: dict[str, Tensor]
        ) -> Tensor:
            return (self.mu / 2.0) * sum(
                (
                    (parameter - reference_parameters[name].to(parameter.device))
                    .square()
                    .sum()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ),
                torch.zeros((), device=next(model.parameters()).device),
            )

    def client_update(self, *, worker: Any, job: Job) -> Result:
        return worker.update(job=job, client=self.Client(mu=self.config.mu))
