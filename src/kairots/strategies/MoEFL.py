"""Specialized federated learning using a mixture of experts, Zec et al. (2020).

Reference: arXiv:2010.02056v3 and edvinli/federated-learning-mixture@ddc9e9b3 (MIT).
Step 1: FedAvg trains the global model on the opt-in clients for every round but the
last. Step 2, in the last round, every client (opt-out ones too) fine-tunes a copy of
the global model it received into a local specialist. Step 3 trains the specialist and
an input-conditioned gate (the expert architecture with one sigmoid output) on the
client's data with the global expert frozen: `h(x) f_s(x) + (1 - h(x)) f_g(x)`.
Both steps run Adam with early stopping on the client's validation windows. The
specialist and gate stay on the client and are never sent; the last round sends no
uplink. Clients are scored with their mixture as personalization.

The paper is a preprint, so `reference: "code"` (the default) follows the official
code where the two disagree and `reference: "paper"` follows the paper. Lines whose
behavior comes from the code are commented `official code:` or `paper vs code:`.
"""

import copy
import math
import random
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from kairots.registry import get
from kairots.runtime.contracts import ClientState, Job, Message, Result
from kairots.strategies.base import Strategy
from kairots.strategies.Traditional import Traditional

REFERENCES = ("code", "paper")


class MoEFL(Traditional):
    TASKS = ("Classification",)

    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "MoEFL"
        reference: str = "code"
        # Fraction q of clients that opt out of FedAvg (they still get the global
        # model and train their mixture on all their data).
        opt_out_fraction: float = 0.0
        opt_out_seed: int = 0
        # official code: 500 epochs for the fine-tune and the mixture (paper agrees).
        specialist_epochs: int = 500
        mixture_epochs: int = 500
        # paper vs code: the paper fine-tunes at 1e-5 (CIFAR-10, Fashion-MNIST); the
        # code hard-codes 5e-5.
        specialist_learning_rate: float = 5e-5
        # official code: the mixture uses `--lr` (the README and the paper: 1e-5).
        mixture_learning_rate: float = 1e-5
        train_gate_only: bool = False
        # official code: validate every 5 epochs; stop after 10 checks without an
        # absolute 0.01 drop in validation loss (the paper says only "early stopping").
        validation_interval: int = 5
        stopping_checks: int = 10
        minimum_improvement: float = 0.01

        def __post_init__(self) -> None:
            if self.reference not in REFERENCES:
                raise ValueError(f"MoEFL reference must be one of {REFERENCES}")
            if not 0.0 <= self.opt_out_fraction < 1.0:
                raise ValueError("MoEFL opt_out_fraction lies in [0, 1)")
            if min(self.specialist_epochs, self.mixture_epochs) < 0:
                raise ValueError("MoEFL epochs must be >= 0")
            if self.validation_interval < 1 or self.stopping_checks < 1:
                raise ValueError(
                    "MoEFL validation_interval and stopping_checks must be >= 1"
                )

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.config: MoEFL.Config = config
        # The gate's initial values, filled when the scoring mixture is built; a
        # client without a mixture is scored with them over two global experts.
        self.gate_template: dict[str, Tensor] = {}

    @staticmethod
    def validate(*, experiment: Any) -> None:
        if getattr(experiment.evaluation, "finetune_epochs", 0):
            raise ValueError(
                "MoEFL scores a mixture; evaluation.finetune_epochs needs one model"
            )
        if experiment.groups or experiment.client_model is not None:
            raise ValueError("MoEFL needs one model architecture for every client")
        if experiment.strategy.iterations < 2:
            raise ValueError("MoEFL needs >= 1 FedAvg round before the mixture round")
        if not hasattr(experiment.model, "num_classes"):
            raise ValueError("MoEFL builds its gate from a model with num_classes")

    class Gate(nn.Module):
        """The expert architecture with one output and a sigmoid (paper Sec. 4)."""

        def __init__(self, *, body: nn.Module) -> None:
            super().__init__()
            self.body = body

        def forward(self, *, batch_x: Tensor) -> Tensor:
            # KairoTS classification heads need two classes; sigmoid of the logit
            # difference spans the same functions as one sigmoid unit after the
            # final linear layer.
            logits = self.body(batch_x=batch_x)
            return torch.sigmoid(logits[..., 1:2] - logits[..., 0:1])

    class Mixture(nn.Module):
        """The gated mixture of the local specialist and the global expert."""

        def __init__(
            self, *, expert: nn.Module, gate: "MoEFL.Gate", reference: str
        ) -> None:
            super().__init__()
            self.global_expert = expert
            self.local_expert = copy.deepcopy(expert)
            self.gate = gate
            self.reference = reference

        def forward(self, *, batch_x: Tensor) -> Tensor:
            """Log-probabilities of the mixture."""
            weight = self.gate(batch_x=batch_x).reshape(-1, 1)
            local = functional.log_softmax(self.local_expert(batch_x=batch_x), dim=-1)
            shared = functional.log_softmax(self.global_expert(batch_x=batch_x), dim=-1)
            # paper vs code: the paper mixes the experts' softmax outputs (eq. 3, Sec.
            # 4); the code mixes their log-probabilities (`train_mix`, `test_img_mix`).
            if self.reference == "paper":
                return torch.log(weight * local.exp() + (1.0 - weight) * shared.exp())
            return weight * local + (1.0 - weight) * shared

    class LogProbabilities(nn.Module):
        """An expert's log-softmax output, as the official networks end."""

        def __init__(self, *, expert: nn.Module) -> None:
            super().__init__()
            self.expert = expert

        def forward(self, *, batch_x: Tensor) -> Tensor:
            return functional.log_softmax(self.expert(batch_x=batch_x), dim=-1)

    def _mixture(self, *, expert: nn.Module, experiment: Any) -> "MoEFL.Mixture":
        model = experiment.model
        gate_config = replace(model, num_classes=2)
        return self.Mixture(
            expert=expert,
            gate=self.Gate(
                body=get(category="models", name=model.name)(config=gate_config)
            ),
            reference=self.config.reference,
        )

    def personal_model(self, *, model: nn.Module, experiment: Any) -> nn.Module:
        mixture = self._mixture(expert=copy.deepcopy(model), experiment=experiment)
        self.gate_template = {
            key: value.detach().clone()
            for key, value in mixture.gate.state_dict().items()
        }
        return mixture

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        mixture: Any
        config: "MoEFL.Config"
        validation: list[tuple[Tensor, Tensor]]
        mixture_state: dict[str, Tensor]

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
            if self.mixture is None:
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
            loss, self.mixture_state = self.fit_mixture(
                batches=batches, device=device, global_parameters=reference_parameters
            )
            return loss

        def _validation_loss(self, *, module: nn.Module, device: Any) -> float:
            module.eval()
            total, samples = 0.0, 0
            with torch.no_grad():
                for batch_x, batch_y in self.validation:
                    output = module(batch_x=batch_x.to(device))
                    batch_y = batch_y.to(device)
                    total += functional.nll_loss(output, batch_y).item() * (
                        1 if self.config.reference == "code" else len(batch_y)
                    )
                    samples += len(batch_y)
            # official code: the batch-mean losses are summed and divided by the
            # sample count (`validate`); paper mode takes the per-sample mean.
            return total / max(samples, 1)

        def _train(
            self,
            *,
            module: nn.Module,
            parameters: list[nn.Parameter],
            learning_rate: float,
            epochs: int,
            batches: Any,
            device: Any,
            keep: list[nn.Module],
        ) -> float:
            """Adam with the official early stopping; returns the last epoch's loss."""
            # positional: the torch.optim learning-rate argument
            optimizer = torch.optim.Adam(parameters, learning_rate)
            best_loss = math.inf
            best_states = [copy.deepcopy(part.state_dict()) for part in keep]
            counter = 0
            epoch_loss = 0.0
            for epoch in range(epochs):
                # official code: every module, the frozen global expert included, is
                # in train mode while the mixture trains (the paper is silent).
                module.train()
                losses = []
                for batch_x, batch_y in batches:
                    output = module(batch_x=batch_x.to(device))
                    loss = functional.nll_loss(output, batch_y.to(device))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                epoch_loss = sum(losses) / max(len(losses), 1)
                if epoch % self.config.validation_interval:
                    continue
                validation_loss = self._validation_loss(module=module, device=device)
                if validation_loss < best_loss - self.config.minimum_improvement:
                    best_loss, counter = validation_loss, 0
                    best_states = [copy.deepcopy(part.state_dict()) for part in keep]
                else:
                    counter += 1
                if counter == self.config.stopping_checks:
                    break
            # paper vs code: the paper returns the best validated model; the code
            # saves `state_dict()` without a copy, so it returns the last weights.
            if self.config.reference == "paper":
                for part, state in zip(keep, best_states, strict=True):
                    part.load_state_dict(state)
            return epoch_loss

        def fit_mixture(
            self, *, batches: Any, device: Any, global_parameters: dict[str, Tensor]
        ) -> tuple[float, dict[str, Tensor]]:
            mixture: MoEFL.Mixture = self.mixture.to(device)
            mixture.global_expert.load_state_dict(global_parameters)
            # Step 2: the specialist starts from the global model (paper and code).
            mixture.local_expert.load_state_dict(global_parameters)
            mixture.global_expert.requires_grad_(False)
            specialist = mixture.local_expert
            self._train(
                module=MoEFL.LogProbabilities(expert=specialist),
                parameters=list(specialist.parameters()),
                learning_rate=self.config.specialist_learning_rate,
                epochs=self.config.specialist_epochs,
                batches=batches,
                device=device,
                keep=[specialist],
            )
            # Step 3: the gate, plus the specialist unless `train_gate_only`.
            parameters = list(mixture.gate.parameters())
            if self.config.train_gate_only:
                specialist.requires_grad_(False)
            else:
                parameters += list(specialist.parameters())
            loss = self._train(
                module=mixture,
                parameters=parameters,
                learning_rate=self.config.mixture_learning_rate,
                epochs=self.config.mixture_epochs,
                batches=batches,
                device=device,
                keep=[mixture.gate, specialist],
            )
            return loss, {
                key: value.detach().cpu().clone()
                for key, value in mixture.state_dict().items()
            }

    def _mixture_round(self, *, current_iteration: int) -> bool:
        return current_iteration % self.config.iterations == self.config.iterations - 1

    def opted_out(self, *, client_ids: list[int]) -> set[int]:
        # official code: int(q * K) clients drawn without replacement.
        count = int(self.config.opt_out_fraction * len(client_ids))
        return set(
            random.Random(self.config.opt_out_seed).sample(sorted(client_ids), count)
        )

    def select_clients(
        self, *, client_ids: list[int], current_iteration: int, seed: int
    ) -> list[int]:
        """FedAvg rounds sample from the opt-in clients, keeping the count the
        join ratio gives over all clients (paper Sec. 4); the mixture round
        selects every client."""
        if self._mixture_round(current_iteration=current_iteration):
            return sorted(client_ids)
        opted_out = self.opted_out(client_ids=client_ids)
        opt_in = sorted(client for client in client_ids if client not in opted_out)
        count = max(1, math.ceil(len(client_ids) * self.config.join_ratio))
        return sorted(random.Random(seed).sample(opt_in, min(count, len(opt_in))))

    @staticmethod
    def _validation(*, worker: Any, client_id: int) -> list[tuple[Tensor, Tensor]]:
        windows = worker.client_data(client_id=client_id).splits.get("validation", [])
        if not windows:
            raise ValueError("MoEFL early stopping needs validation windows")
        experiment = worker.client_experiment(client_id=client_id)
        input_length, output_length, offset_length = experiment.lengths()
        task, view, _ = worker.components(client_id=client_id)
        return [
            task.prepare(
                batch_x=batch_x,
                batch_y=batch_y,
                output_length=output_length,
                generator=torch.Generator(),
            )
            for batch_x, batch_y in view.batches(
                values=worker.client_data(client_id=client_id).values,
                windows=windows,
                input_length=input_length,
                output_length=output_length,
                offset_length=offset_length,
                batch_size=experiment.data.source.batch_size,
                shuffle=False,
                generator=torch.Generator(),
            )
        ]

    def client_update(self, *, worker: Any, job: Job) -> Result:
        if not self._mixture_round(current_iteration=job.operation_id[0]):
            return worker.update(
                job=job,
                client=self.Client(
                    mixture=None, config=self.config, validation=[], mixture_state={}
                ),
            )
        experiment = worker.client_experiment(client_id=job.client_id)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(job.seed)
            mixture = self._mixture(
                expert=get(category="models", name=experiment.model.name)(
                    config=experiment.model
                ),
                experiment=experiment,
            )
        client = self.Client(
            mixture=mixture,
            config=self.config,
            validation=self._validation(worker=worker, client_id=job.client_id),
            mixture_state={},
        )
        result: Result = worker.update(job=job, client=client)
        result.state.extra = {**result.state.extra, "mixture": client.mixture_state}
        # The mixture round sends nothing back; the global model is final.
        result.uplink = Message(
            sender=result.uplink.sender, receiver=result.uplink.receiver
        )
        return result

    def client_parameters(
        self, *, global_params: dict[str, Tensor], state: ClientState
    ) -> dict[str, Tensor]:
        """The client's trained mixture; before it has one, the global model as
        both experts (any gate weight then gives the global prediction)."""
        if "mixture" in state.extra:
            return state.extra["mixture"]
        return {
            **{f"global_expert.{key}": value for key, value in global_params.items()},
            **{f"local_expert.{key}": value for key, value in global_params.items()},
            **{f"gate.{key}": value for key, value in self.gate_template.items()},
        }
