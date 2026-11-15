"""SCAFFOLD: stochastic controlled averaging, Karimireddy et al. (ICML 2020).

The server keeps the model `x` and a control variate `c`; each client keeps its own
`c_i`, all zero at the start. A selected client receives `(x, c)`, starts from
`y = x` and takes `K` plain SGD steps `y <- y - lr * (g(y) - c_i + c)` (`K` is its
epochs times its batches). It then sets `c_i+` to the gradient of its train loss at
`x` (option `I`, one more pass over its batches) or to
`c_i - c + (x - y) / (K * lr)` (option `II`) and uploads `(y - x, c_i+ - c_i)`. The
server takes `x <- x + global_step_size * mean(y - x)` and
`c <- c + sum(c_i+ - c_i) / N`, with `N` the clients it selects from (Algorithm 1).
Control variates cover the model's parameters, so a model with buffers is refused;
the local steps need the `SGD` optimizer without momentum or weight decay.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

from kairots.runtime.contracts import Job, Message, Result
from kairots.strategies.base import Strategy
from kairots.strategies.Traditional import Traditional

# Prefix of the control-variate tensors in the downlink and the uplink.
CONTROL = "control."


class SCAFFOLD(Traditional):
    REQUIRES = {
        "learner.optimizer.name": ("SGD",),
        "learner.optimizer.momentum": (0.0,),
        "learner.optimizer.weight_decay": (0.0,),
    }

    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "SCAFFOLD"
        global_step_size: float = 1.0
        control_update: str = "II"

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.config: SCAFFOLD.Config = config
        if config.control_update not in ("I", "II"):
            raise ValueError("SCAFFOLD control_update is 'I' or 'II'")
        self.control: dict[str, Tensor] = {}
        self.population = 0

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        server: dict[str, Tensor]
        own: dict[str, Tensor]
        control_update: str
        updated: dict[str, Tensor] = field(default_factory=dict)

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
                raise ValueError(
                    "SCAFFOLD's control variates cover parameters; the model has "
                    "buffers"
                )
            parameters = dict(model.named_parameters())
            if self.control_update == "I":
                model.train()
                model.zero_grad(set_to_none=True)
                for batch_x, batch_y in batches:
                    loss = loss_function(
                        prediction=model(batch_x=batch_x.to(device)),
                        target=batch_y.to(device),
                    )
                    (loss / len(batches)).backward()
                self.updated = {
                    name: torch.zeros_like(self.own[name])
                    if parameter.grad is None
                    else parameter.grad.detach().cpu().clone()
                    for name, parameter in parameters.items()
                }
                model.zero_grad(set_to_none=True)
            corrections = {
                name: (self.server[name] - self.own[name]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                for name, parameter in parameters.items()
            }
            # Option II divides the displacement by the summed step sizes, K * lr
            # under a constant rate.
            rates: list[float] = []
            step_hook = optimizer.register_step_pre_hook(
                lambda *_: rates.append(optimizer.param_groups[0]["lr"])
            )
            handles = [
                parameter.register_hook(
                    lambda grad, correction=corrections[name]: grad + correction
                )
                for name, parameter in parameters.items()
                if parameter.requires_grad
            ]
            try:
                loss = super().fit(
                    learner=learner,
                    model=model,
                    optimizer=optimizer,
                    batches=batches,
                    epochs=epochs,
                    device=device,
                    loss_function=loss_function,
                    reference_parameters=reference_parameters,
                )
            finally:
                step_hook.remove()
                for handle in handles:
                    handle.remove()
            if self.control_update == "II":
                scale = math.fsum(rates)
                self.updated = {
                    name: self.own[name]
                    - self.server[name]
                    + (reference_parameters[name] - parameter.detach().cpu()) / scale
                    for name, parameter in parameters.items()
                }
            return loss

    def select_clients(
        self, *, client_ids: list[int], current_iteration: int, seed: int
    ) -> list[int]:
        self.population = len(client_ids)
        return super().select_clients(
            client_ids=client_ids, current_iteration=current_iteration, seed=seed
        )

    def build_downlink(
        self, *, global_params: dict[str, Tensor], client_id: int
    ) -> Message:
        if not self.control:
            self.control = {
                key: torch.zeros_like(value) for key, value in global_params.items()
            }
        return Message(
            sender="server",
            receiver=f"client:{client_id}",
            tensors={
                **global_params,
                **{CONTROL + key: value for key, value in self.control.items()},
            },
        )

    def client_update(self, *, worker: Any, job: Job) -> Result:
        tensors = job.downlink.tensors
        server = {
            key.removeprefix(CONTROL): value
            for key, value in tensors.items()
            if key.startswith(CONTROL)
        }
        start = {
            key: value for key, value in tensors.items() if not key.startswith(CONTROL)
        }
        client = self.Client(
            server=server,
            own=job.state.extra.get("control")
            or {key: torch.zeros_like(value) for key, value in server.items()},
            control_update=self.config.control_update,
        )
        result: Result = worker.update(
            job=replace(job, downlink=replace(job.downlink, tensors=start)),
            client=client,
        )
        result.uplink.tensors = {
            **{key: value - start[key] for key, value in result.uplink.tensors.items()},
            **{
                CONTROL + key: value - client.own[key]
                for key, value in client.updated.items()
            },
        }
        result.state.extra = {**result.state.extra, "control": client.updated}
        return result

    def uplink_scalars(
        self, *, tensors: dict[str, Tensor], train_samples: int
    ) -> dict[str, float]:
        """Unweighted averaging: no sample count is sent."""
        return {}

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """`x + global_step_size * mean(y - x)` and `c + sum(c_i+ - c_i) / N`."""
        self.control = {
            key: value
            + sum(message.tensors[CONTROL + key] for message in uplinks)
            / self.population
            for key, value in self.control.items()
        }
        return {
            key: value
            + self.config.global_step_size
            * sum(message.tensors[key] for message in uplinks)
            / len(uplinks)
            for key, value in global_params.items()
        }
