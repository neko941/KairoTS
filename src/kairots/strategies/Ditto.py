"""Ditto: Fair and Robust Federated Learning Through Personalization.

Reference: Li et al., ICML 2021, arXiv:2012.04221v3, Section 3 and Algorithm 1.

Each round a selected client first trains its personal model with the proximal
term `mu / 2 * ||personal - global||^2`, with plain SGD at the learner's learning
rate for `personal_epochs` epochs, then trains the global model as FedAvg does.
"""

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.optimizers.SGD import SGD
from kairots.runtime.contracts import Job, Result
from kairots.strategies.base import Strategy
from kairots.strategies.Personalized import Personalized
from kairots.strategies.Proximal import Proximal


class Ditto(Personalized):
    """Ditto leaf strategy for personalized federated learning."""

    @dataclass(kw_only=True)
    class Config(Personalized.Config, Proximal.Config):
        name: str = "Ditto"
        mu: float = 0.1
        personal_epochs: int = 1

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.config: Ditto.Config = config

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        personal: dict[str, Tensor]
        mu: float
        personal_epochs: int

        def personal_proximal_term(
            self, *, model: nn.Module, reference_parameters: dict[str, Tensor]
        ) -> Tensor:
            term = torch.zeros((), device=next(model.parameters()).device)
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or name not in reference_parameters:
                    continue
                reference = reference_parameters[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
                term = term + (parameter - reference).square().sum()
            return (self.mu / 2.0) * term

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
            if self.personal_epochs > 0:
                personal_model = copy.deepcopy(model)
                personal_model.load_state_dict(self.personal)
                # (torch's own keyword name, so it goes through a dict.)
                personal_optimizer = SGD(config=SGD.Config()).build(
                    parameters=personal_model.parameters(),
                    learning_rate=learner.config.learning_rate,
                    fused=False,
                )
                for _ in range(self.personal_epochs):
                    learner.fit_epoch(
                        model=personal_model,
                        optimizer=personal_optimizer,
                        batches=batches,
                        device=device,
                        loss_function=loss_function,
                        regularizer=lambda: self.personal_proximal_term(
                            model=personal_model,
                            reference_parameters=reference_parameters,
                        ),
                    )
                self.personal = {
                    key: value.detach().cpu().clone()
                    for key, value in personal_model.state_dict().items()
                }
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

    def client_update(self, *, worker: Any, job: Job) -> Result:
        client = self.Client(
            personal=job.state.extra["personal"],
            mu=self.config.mu,
            personal_epochs=self.config.personal_epochs,
        )
        result: Result = worker.update(job=job, client=client)
        result.state.extra = {**result.state.extra, "personal": client.personal}
        return result
