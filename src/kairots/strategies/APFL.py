"""APFL: adaptive personalized federated learning, Deng et al. (2020).

Each client keeps a local model `v` (initialized from the global model) and a mixing
weight `alpha`. On every batch a selected client steps the global model `w` as
FedAvg does and steps `v` along the gradient of the loss at the mixed model
`alpha * v + (1 - alpha) * w`, both from the parameters before the step (Algorithm
1). With `adaptive_alpha`, the first step of each round also moves `alpha` along
`<v - w, gradient at the mixed model>` and clips it to [0, 1] (eq. alpha_gd). `v`
and `alpha` take plain SGD steps at the learner's learning rate. The server takes
the unweighted mean of the selected clients' `w`, and each client is scored with
`alpha * v + (1 - alpha) * w`. A model with buffers is refused, since the mixed
forward pass would update the global model's buffers.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.func import functional_call

from kairots.runtime.contracts import ClientState, Job, Message, Result
from kairots.strategies.base import Strategy
from kairots.strategies.Personalized import Personalized


class APFL(Personalized):
    @dataclass(kw_only=True)
    class Config(Personalized.Config):
        name: str = "APFL"
        alpha: float = 0.5
        adaptive_alpha: bool = True

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.config: APFL.Config = config
        if not 0.0 <= config.alpha <= 1.0:
            raise ValueError("APFL alpha lies in [0, 1]")

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        personal: dict[str, Tensor]
        alpha: float
        adaptive_alpha: bool

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
            if next(model.buffers(), None) is not None:
                raise ValueError("APFL mixes parameters; the model has buffers")
            learning_rate = learner.config.learning_rate
            personal = {
                name: value.detach().clone().to(device).requires_grad_()
                for name, value in self.personal.items()
            }
            alpha = self.alpha
            losses: list[float] = []
            for epoch in range(epochs):
                losses = []
                for step, (batch_x, batch_y) in enumerate(batches):
                    weight = torch.tensor(alpha, device=device, requires_grad=True)
                    mixed = {
                        name: weight * personal[name]
                        + (1.0 - weight) * parameter.detach()
                        for name, parameter in model.named_parameters()
                    }
                    loss = loss_function(
                        prediction=functional_call(
                            model, mixed, kwargs={"batch_x": batch_x.to(device)}
                        ),
                        target=batch_y.to(device),
                    )
                    *gradients, alpha_gradient = torch.autograd.grad(
                        loss, [*personal.values(), weight]
                    )
                    losses.append(
                        learner.fit_epoch(
                            model=model,
                            optimizer=optimizer,
                            batches=[(batch_x, batch_y)],
                            device=device,
                            loss_function=loss_function,
                            regularizer=lambda: self.proximal_term(
                                model=model, reference_parameters=reference_parameters
                            ),
                        )
                    )
                    with torch.no_grad():
                        for value, gradient in zip(personal.values(), gradients):
                            value -= learning_rate * gradient
                    if self.adaptive_alpha and epoch == 0 and step == 0:
                        alpha = min(
                            max(alpha - learning_rate * alpha_gradient.item(), 0.0),
                            1.0,
                        )
            self.personal = {
                name: value.detach().cpu().clone() for name, value in personal.items()
            }
            self.alpha = alpha
            return sum(losses) / max(len(losses), 1)

    def initial_state(self, *, global_params: dict[str, Tensor]) -> ClientState:
        state = super().initial_state(global_params=global_params)
        state.extra["alpha"] = self.config.alpha
        return state

    def client_parameters(
        self, *, global_params: dict[str, Tensor], state: ClientState
    ) -> dict[str, Tensor]:
        alpha = state.extra["alpha"]
        return {
            key: alpha * state.extra["personal"][key] + (1.0 - alpha) * value
            for key, value in global_params.items()
        }

    def client_update(self, *, worker: Any, job: Job) -> Result:
        client = self.Client(
            personal=job.state.extra["personal"],
            alpha=job.state.extra["alpha"],
            adaptive_alpha=self.config.adaptive_alpha,
        )
        result: Result = worker.update(job=job, client=client)
        result.state.extra = {
            **result.state.extra,
            "personal": client.personal,
            "alpha": client.alpha,
        }
        return result

    def uplink_scalars(
        self, *, tensors: dict[str, Tensor], train_samples: int
    ) -> dict[str, float]:
        """Unweighted averaging: no sample count is sent."""
        return {}

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Unweighted mean of the selected clients' global models."""
        return {
            key: torch.stack([message.tensors[key] for message in uplinks]).mean(dim=0)
            for key in global_params
        }
