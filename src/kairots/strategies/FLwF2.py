"""FLwF-2, Usmanova et al. (SmartComp 2022), two-teacher classification distillation.

The client teacher is its previous communication-round model. Both teachers are
frozen; only the student's full parameters are sent. The paper's losses sum over
samples and classes, without a temperature-squared multiplier.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from kairots.strategies.Traditional import Traditional


class FLwF2(Traditional):
    config: "FLwF2.Config"
    TASKS = ("Classification",)
    REQUIRES = {
        "training.name": ("ClassIncremental",),
        "learner.name": ("Gradient",),
        "learner.optimizer.name": ("SGD",),
        "learner.precision": ("fp32",),
    }

    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "FLwF2"
        alpha: float = 0.001
        beta: float = 0.7
        temperature: float = 2.0

        def __post_init__(self) -> None:
            if not (0 <= self.alpha <= 1 and 0 <= self.beta <= 1 - self.alpha):
                raise ValueError("FLwF2 needs alpha, beta >= 0 and alpha + beta <= 1")
            if not self.temperature > 0:
                raise ValueError("FLwF2 temperature must be positive")
            if self.join_ratio != 1:
                raise ValueError("FLwF2 Algorithm 1 uses every client each round")

    @staticmethod
    def objective(
        *,
        prediction: Tensor,
        target: Tensor,
        server: Tensor,
        previous: Tensor | None,
        config: Config,
    ) -> Tensor:
        """The three summed cross entropies in the paper's final loss equation."""
        loss = config.alpha * functional.cross_entropy(
            prediction, target, reduction="sum"
        )
        log_probabilities = functional.log_softmax(
            prediction / config.temperature, dim=-1
        )
        for weight, teacher in (
            (config.beta, previous),
            (1 - config.alpha - config.beta, server),
        ):
            if teacher is not None and weight:
                probabilities = functional.softmax(
                    teacher.detach() / config.temperature, dim=-1
                )
                loss = loss - weight * (probabilities * log_probabilities).sum()
        return loss

    @dataclass(kw_only=True)
    class Client(Traditional.Client):
        config: "FLwF2.Config"
        previous: dict[str, Tensor] | None

        def fit(
            self,
            *,
            learner: Any,
            model: nn.Module,
            optimizer: Any,
            batches: Any,
            epochs: int,
            device: Any,
            loss_function: Any,
            reference_parameters: dict[str, Tensor],
        ) -> float:
            server = deepcopy(model).eval().requires_grad_(False)
            previous = None
            if self.previous is not None:
                previous = deepcopy(server)
                previous.load_state_dict(self.previous)
            total, count = 0.0, 0
            model.train()
            for _ in range(epochs):
                for batch_x, batch_y in batches:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    with torch.no_grad():
                        server_scores = server(batch_x=batch_x)
                        previous_scores = (
                            None if previous is None else previous(batch_x=batch_x)
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss = FLwF2.objective(
                        prediction=model(batch_x=batch_x),
                        target=batch_y,
                        server=server_scores,
                        previous=previous_scores,
                        config=self.config,
                    )
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())
                    count += batch_y.numel()
            return total / max(count, 1)

    def client_update(self, *, worker: Any, job: Any) -> Any:
        result = worker.update(
            job=job,
            client=self.Client(
                config=self.config, previous=job.state.extra.get("previous")
            ),
        )
        result.state.extra["previous"] = {
            key: value.clone() for key, value in result.uplink.tensors.items()
        }
        return result

    def client_parameters(
        self, *, global_params: dict[str, Tensor], state: Any
    ) -> dict[str, Tensor]:
        return state.extra.get("previous", global_params)
