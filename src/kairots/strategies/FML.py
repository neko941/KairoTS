"""Federated Mutual Learning (Shen et al. (2020)).

Each client trains a private classifier and the shared meme classifier together.
Their supervised losses are mixed with KL distillation over class probabilities.
The meme optimizer's per-parameter state restarts each round and keeps its
scheduled learning rate, which the persistent private optimizer copies; clients are
scored with their private model.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from kairots.registry import get
from kairots.runtime.contracts import ClientState, Job, Message, Result
from kairots.strategies.base import Strategy
from kairots.strategies.Heterogeneous import Heterogeneous


class FML(Heterogeneous):
    TASKS = ("Classification",)
    CLIENT_MODEL = True

    @dataclass(kw_only=True)
    class Config(Heterogeneous.Config):
        name: str = "FML"
        alpha: float = 0.5
        beta: float = 0.5

        def __post_init__(self) -> None:
            if not 0.0 <= self.alpha <= 1.0 or not 0.0 <= self.beta <= 1.0:
                raise ValueError("FML alpha and beta must lie in [0, 1]")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        personal: dict[str, Tensor]
        personal_optimizer: dict[str, Any]
        private_config: Any
        alpha: float
        beta: float

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
            private = get(category="models", name=self.private_config.name)(
                config=self.private_config
            ).to(device)
            if self.personal:
                private.load_state_dict(self.personal)
            private_optimizer = learner.optimizer(model=private)
            if self.personal_optimizer:
                private_optimizer.load_state_dict(self.personal_optimizer)
            for private_group, meme_group in zip(
                private_optimizer.param_groups, optimizer.param_groups, strict=True
            ):
                private_group["lr"] = meme_group["lr"]
            private.train()
            model.train()
            losses: list[float] = []
            for _ in range(epochs):
                losses = []
                for batch_x, batch_y in batches:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    output = private(batch_x=batch_x)
                    output_g = model(batch_x=batch_x)
                    loss = self.alpha * loss_function(
                        prediction=output, target=batch_y
                    ) + (1.0 - self.alpha) * functional.kl_div(
                        input=functional.log_softmax(output, dim=1),
                        target=functional.softmax(output_g.detach(), dim=1),
                        reduction="batchmean",
                    )
                    loss_g = self.beta * loss_function(
                        prediction=output_g, target=batch_y
                    ) + (1.0 - self.beta) * functional.kl_div(
                        input=functional.log_softmax(output_g, dim=1),
                        target=functional.softmax(output.detach(), dim=1),
                        reduction="batchmean",
                    )
                    private_optimizer.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(private.parameters(), 10)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    private_optimizer.step()
                    optimizer.step()
                    losses.append(loss.item())
            self.personal = {
                key: value.detach().cpu().clone()
                for key, value in private.state_dict().items()
            }
            self.personal_optimizer = private_optimizer.state_dict()
            return sum(losses) / max(len(losses), 1)

    def select_clients(
        self, *, client_ids: list[int], current_iteration: int, seed: int
    ) -> list[int]:
        return sorted(client_ids)

    def initial_state(self, *, global_params: dict[str, Tensor]) -> ClientState:
        return ClientState()

    def client_parameters(
        self, *, global_params: dict[str, Tensor], state: ClientState
    ) -> dict[str, Tensor]:
        return state.extra.get("personal", global_params)

    def client_update(self, *, worker: Any, job: Job) -> Result:
        config = worker.client_experiment(client_id=job.client_id)
        personal = job.state.extra.get("personal", {})
        if not personal and config.client_model is None:
            personal = job.downlink.tensors
        client = self.Client(
            personal=personal,
            personal_optimizer=job.state.extra.get("personal_optimizer", {}),
            private_config=config.client_model or config.model,
            alpha=self.config.alpha,
            beta=self.config.beta,
        )
        result = worker.update(job=job, client=client)
        if result.state.optimizer_state:
            result.state.optimizer_state = {
                "state": {},
                "param_groups": result.state.optimizer_state["param_groups"],
            }
        result.state.extra = {
            **result.state.extra,
            "personal": client.personal,
            "personal_optimizer": client.personal_optimizer,
        }
        return result

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        return {
            key: torch.stack([message.tensors[key] for message in uplinks]).mean(dim=0)
            for key in global_params
        }
