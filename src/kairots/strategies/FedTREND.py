"""Fed-TREND: synthetic data against heterogeneity in federated forecasting, Yuan et
al. (2024).

Reference: arXiv:2411.15716v1 and hi-weiyuan/FedTREND@54db4c8f (the authors' "draft
code"). The server builds two synthetic datasets by trajectory matching (MTT):

- `D_ct` from the clients' uploaded models, every `client_interval` rounds; it is sent
  to every client (a billed downlink tensor) and mixed into local training.
- `D_gt` from the aggregated global models, every `global_interval` rounds; it stays
  on the server and fine-tunes (refines) the aggregated global model.

Each synthetic sample is one univariate series of `input_length + output_length` steps
whose head is the input and whose tail the target, so Fed-TREND runs on univariate
clients (one channel per client: `RowConcat`, `Collection`), as the paper's devices.

The paper is a preprint, so `reference: "code"` (the default) follows the official
code where the two disagree and `reference: "paper"` follows the paper; details the
paper leaves open come from the code in both modes. Lines whose behavior comes from
the code are commented `official code:` or `paper vs code:`.
"""

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional

from kairots.registry import get
from kairots.runtime.contracts import Job, Message, Result
from kairots.runtime.device import execution_device
from kairots.runtime.seeds import derive_seed
from kairots.strategies.base import Strategy
from kairots.strategies.Traditional import Traditional

REFERENCES = ("code", "paper")
# Downlink key of D_ct; the client strips it before loading the model.
SYNTHETIC = "fedtrend.synthetic"


class FedTREND(Traditional):
    TASKS = ("Forecasting",)
    SOURCES = ("RowConcat", "Collection")
    # MTT unrolls gradient steps of the model, so clients train by gradient.
    REQUIRES = {"learner.name": ("Gradient",)}

    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "FedTREND"
        reference: str = "code"
        # Ablation switches (paper Table 3): consistent updating ("CU"), D_ct, D_gt.
        consistency: bool = True
        client_synthetic: bool = True
        global_refinement: bool = True
        # L_ct and L_gt, rounds between constructions (paper and code: 10).
        client_interval: int = 10
        global_interval: int = 10
        # |D_ct| (paper Sec. 3.6 and code: 20 pairs). |D_gt| 0 is one series per
        # client (official code: `--N -1`; the paper does not state it).
        client_synthetic_size: int = 20
        global_synthetic_size: int = 0
        # official code: an update element is kept while its sign agrees with the
        # running frequency of its past signs by more than this threshold.
        consistency_threshold: float = 0.3
        # official code (the paper is silent): MTT inner loop, SGD steps of the
        # student on `synthetic_batch_size` synthetic pairs, batch_y blended with
        # the target model's prediction (`alpha`, soft labels).
        synthetic_steps: int = 20
        synthetic_batch_size: int = 64
        inner_learning_rate: float = 3e-4
        soft_label_ratio: float = 0.2
        # paper vs code: Adam on the synthetic series at 0.1 (code `lr_feat`); the
        # paper gives 3e-4. 0 picks the value of `reference`.
        synthetic_learning_rate: float = 0.0
        # Outer MTT iterations (paper: 300; official code: 300 + 1), each matching
        # one sampled segment `multistep` times (official code).
        construction_iterations: int = 300
        multistep: int = 2
        # official code: D_gt segments span 2 rounds (`expert_epochs`).
        global_segment: int = 2
        # official code: the global model is refined in the 8 rounds after each D_gt
        # construction, one SGD pass at the local learning rate and momentum on
        # batches of 64; paper vs code: the paper refines in every round.
        refinement_rounds: int = 8
        refinement_epochs: int = 1
        refinement_batch_size: int = 64
        refinement_learning_rate: float = 5e-4
        refinement_momentum: float = 0.9

        def __post_init__(self) -> None:
            if self.reference not in REFERENCES:
                raise ValueError(f"FedTREND reference must be one of {REFERENCES}")
            if min(self.client_interval, self.global_interval) < 1:
                raise ValueError("FedTREND intervals must be >= 1")
            if self.global_segment < 1:
                raise ValueError("FedTREND global_segment must be >= 1")
            # official code: segment starts are drawn below interval - segment.
            if self.global_interval <= self.global_segment:
                raise ValueError("FedTREND global_interval must exceed global_segment")
            if self.client_synthetic_size < 1 or self.global_synthetic_size < 0:
                raise ValueError("FedTREND synthetic sizes must be positive")
            if not 0.0 <= self.consistency_threshold <= 1.0:
                raise ValueError("FedTREND consistency_threshold lies in [0, 1]")
            if (
                min(
                    self.synthetic_steps,
                    self.synthetic_batch_size,
                    self.multistep,
                    self.refinement_batch_size,
                )
                < 1
                or min(
                    self.construction_iterations,
                    self.refinement_rounds,
                    self.refinement_epochs,
                )
                < 0
            ):
                raise ValueError("FedTREND step, batch and round counts must be valid")
            if self.synthetic_learning_rate == 0.0:
                self.synthetic_learning_rate = 0.1 if self.reference == "code" else 3e-4

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.config: FedTREND.Config = config
        self.current_iteration = 0
        # Bound by `setup`.
        self.model_config: Any = None
        self.lengths = (0, 0)
        self.seed = 0
        self.device_policy: Any = None
        self.num_gpus = 0
        # Server state, kept across rounds and checkpointed with the strategy.
        self.initial: Tensor | None = None
        self.client_ids: set[int] = set()
        self.global_bank: list[Tensor] = []
        self.client_start: dict[int, Tensor] = {}
        self.sign_frequency: dict[int, dict[str, Tensor]] = {}
        self.updates_seen: dict[int, int] = {}
        self.previous_update: dict[int, dict[str, Tensor]] = {}
        self.client_synthetic: Tensor | None = None
        self.client_version = 0
        self.delivered: dict[int, int] = {}
        self.global_synthetic: Tensor | None = None
        self.refined = 0
        self._module: nn.Module | None = None

    @staticmethod
    def validate(*, experiment: Any) -> None:
        if experiment.groups or experiment.client_model is not None:
            raise ValueError("FedTREND needs one model architecture for every client")
        if experiment.lengths()[2]:
            raise ValueError(
                "FedTREND synthetic series join input and target; offset_length is 0"
            )

    def setup(self, *, experiment: Any) -> None:
        self.model_config = experiment.model
        input_length, output_length, _ = experiment.lengths()
        self.lengths = (input_length, output_length)
        self.seed = experiment.runtime.seed
        self.device_policy = experiment.runtime.device
        self.num_gpus = experiment.runtime.num_gpus

    # Client side.

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        synthetic: Tensor | None
        input_length: int
        batch_size: int
        seed: int

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
            if self.synthetic is None:
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
            synthetic_x = self.synthetic[:, : self.input_length, None]
            synthetic_y = self.synthetic[:, self.input_length :, None]
            generator = torch.Generator().manual_seed(self.seed)
            loss = 0.0
            for _ in range(epochs):
                # official code: D_ct pairs join the local windows in one shuffled
                # dataset, batched as the local data (`rebuild_data_with_syn`).
                local = list(batches)
                pooled_x = torch.cat([batch_x for batch_x, _ in local] + [synthetic_x])
                pooled_y = torch.cat([batch_y for _, batch_y in local] + [synthetic_y])
                order = torch.randperm(len(pooled_x), generator=generator)
                mixed = [
                    (pooled_x[part], pooled_y[part])
                    for part in order.split(self.batch_size)
                ]
                loss = learner.fit_epoch(
                    model=model,
                    optimizer=optimizer,
                    batches=mixed,
                    device=device,
                    loss_function=loss_function,
                    regularizer=lambda: self.proximal_term(
                        model=model, reference_parameters=reference_parameters
                    ),
                )
            return loss

    def build_downlink(
        self, *, global_params: dict[str, Tensor], client_id: int
    ) -> Message:
        """The global model, plus D_ct when the client lacks the latest one (the
        round after each construction under full participation)."""
        message = super().build_downlink(
            global_params=global_params, client_id=client_id
        )
        if (
            self.client_synthetic is not None
            and self.delivered.get(client_id) != self.client_version
        ):
            message.tensors = {**global_params, SYNTHETIC: self.client_synthetic}
            self.delivered[client_id] = self.client_version
        return message

    def uplink_scalars(
        self, *, tensors: dict[str, Tensor], train_samples: int
    ) -> dict[str, float]:
        # paper vs code: the paper averages sample-weighted (eq. 4); the code
        # averages the updates unweighted (`aggregate_gradients`), so sends no count.
        if self.config.reference == "paper":
            return super().uplink_scalars(tensors=tensors, train_samples=train_samples)
        return {}

    def client_update(self, *, worker: Any, job: Job) -> Result:
        tensors = dict(job.downlink.tensors)
        received = tensors.pop(SYNTHETIC, None)
        synthetic = received if received is not None else job.state.extra.get(SYNTHETIC)
        experiment = worker.client_experiment(client_id=job.client_id)
        result: Result = worker.update(
            job=replace(job, downlink=replace(job.downlink, tensors=tensors)),
            client=self.Client(
                synthetic=synthetic,
                input_length=experiment.lengths()[0],
                batch_size=experiment.data.source.batch_size,
                seed=job.seed,
            ),
        )
        if received is not None:
            # The client keeps D_ct until the next one arrives; never billed again.
            result.state.extra = {**result.state.extra, SYNTHETIC: received}
        return result

    # Server side.

    def _device(self) -> torch.device:
        return execution_device(
            policy=self.device_policy, remote=False, num_gpus=self.num_gpus
        )

    def _server_module(self) -> nn.Module:
        module = self._module
        if module is None:
            if self.model_config is None:
                raise RuntimeError("FedTREND.setup must run before aggregation")
            with torch.random.fork_rng(devices=[]):
                module = get(category="models", name=self.model_config.name)(
                    config=self.model_config
                )
            self._module = module
        return module.to(self._device())

    def _names(self) -> list[tuple[str, torch.Size]]:
        return [
            (name, parameter.shape)
            for name, parameter in self._server_module().named_parameters()
        ]

    def _flat(self, *, state: dict[str, Tensor]) -> Tensor:
        """The trainable parameters as one vector (official code: `ReparamModule`)."""
        return torch.cat(
            [
                state[name].detach().reshape(-1).float().cpu()
                for name, _ in self._names()
            ]
        )

    def _forward(self, *, flat: Tensor, batch_x: Tensor) -> Tensor:
        module = self._server_module()
        parameters, offset = {}, 0
        for name, shape in self._names():
            size = math.prod(shape)
            parameters[name] = flat[offset : offset + size].view(shape)
            offset += size
        buffers = dict(module.named_buffers())
        return functional_call(
            module, {**buffers, **parameters}, args=(), kwargs={"batch_x": batch_x}
        )

    def _generator(self, *, operation: str) -> torch.Generator:
        return torch.Generator().manual_seed(
            derive_seed(
                seed=self.seed,
                current_iteration=self.current_iteration,
                client_id=-1,
                operation=operation,
            )
        )

    def construct(
        self,
        *,
        size: int,
        segment: Any,
        distance: Any,
        generator: torch.Generator,
    ) -> Tensor:
        """MTT: learn `size` synthetic series so that training a segment's start on
        them reaches its end. `segment()` draws (start, end, mask) flat vectors;
        `distance(student, start, end, mask)` is the matching loss."""
        device = self._device()
        input_length, output_length = self.lengths
        module = self._server_module()
        # official code: the student model is in train mode while it is matched.
        module.train()
        # official code: series start from a standard normal (`SynData`).
        series = (
            torch.randn(size, input_length + output_length, generator=generator)
            .to(device)
            .requires_grad_(True)
        )
        # positional: the torch.optim learning-rate argument
        optimizer = torch.optim.Adam([series], self.config.synthetic_learning_rate)
        # paper vs code: the code runs `range(0, Iteration + 1)` iterations.
        iterations = self.config.construction_iterations + (
            self.config.reference == "code"
        )
        for _ in range(iterations):
            start, end, mask = (value.to(device) for value in segment())
            for _ in range(self.config.multistep):
                # official code: the synthetic inputs carry gradient, the targets
                # are detached (`syn_Y_total.clone().detach()`); one pair per
                # series, the code's windowing when input and output lengths match.
                synthetic_x = series[:, :input_length, None]
                synthetic_y = series[:, input_length:, None].detach()
                student = start.clone().requires_grad_(True)
                for _ in range(self.config.synthetic_steps):
                    pick = torch.randperm(size, generator=generator)[
                        : self.config.synthetic_batch_size
                    ].to(device)
                    batch_x = synthetic_x[pick]
                    # official code: soft labels blend the stored targets with the
                    # end model's prediction.
                    ratio = self.config.soft_label_ratio
                    soft = synthetic_y[pick] * ratio + self._forward(
                        flat=end, batch_x=batch_x
                    ) * (1.0 - ratio)
                    loss = functional.mse_loss(
                        self._forward(flat=student, batch_x=batch_x), soft
                    )
                    (gradient,) = torch.autograd.grad(loss, student, create_graph=True)
                    student = student - self.config.inner_learning_rate * gradient
                optimizer.zero_grad(set_to_none=True)
                distance(student, start, end, mask).backward()
                optimizer.step()
        return series.detach().cpu()

    @staticmethod
    def global_distance(  # positional: called as distance(student, start, end, mask)
        student: Tensor, start: Tensor, end: Tensor, _: Tensor
    ) -> Tensor:
        # official code (the paper is silent on d): squared distance to the end,
        # over the start's, then divided by its own value (`server_MTT`).
        loss = functional.mse_loss(student, end, reduction="sum") / functional.mse_loss(
            start, end, reduction="sum"
        ).clamp_min(torch.finfo(start.dtype).tiny)
        return loss / loss.item()

    @staticmethod
    def client_distance(  # positional: called as distance(student, start, end, mask)
        student: Tensor, start: Tensor, end: Tensor, mask: Tensor
    ) -> Tensor:
        # official code (the paper is silent on d): cosine distance to the end model
        # (`client_MTT`). Paper mode masks inconsistent elements out of it.
        return 1.0 - functional.cosine_similarity(student * mask, end * mask, dim=0)

    def refine(self, *, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Fine-tune the aggregated global model on D_gt."""
        assert self.global_synthetic is not None
        device = self._device()
        module = self._server_module()
        module.load_state_dict(state)
        module.train()
        # positional: the torch.optim learning-rate argument
        optimizer = torch.optim.SGD(
            module.parameters(),
            self.config.refinement_learning_rate,
            momentum=self.config.refinement_momentum,
        )
        input_length = self.lengths[0]
        series = self.global_synthetic.to(device)
        generator = self._generator(operation="fedtrend-refine")
        for _ in range(self.config.refinement_epochs):
            order = torch.randperm(len(series), generator=generator).to(device)
            for part in order.split(self.config.refinement_batch_size):
                loss = functional.mse_loss(
                    module(batch_x=series[part, :input_length, None]),
                    series[part, input_length:, None],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }

    def sign_mask(
        self, *, client_id: int, update: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """official code (`consistency_mask`): keep an element while the fraction of
        past updates with its current sign exceeds the threshold, then fold this
        update's signs into that running frequency. The first update is kept whole.
        The code counts rounds; under its full participation that is the client's
        update count used here."""
        seen = self.updates_seen.get(client_id, 0)
        rising = {key: (value >= 0).float() for key, value in update.items()}
        if seen == 0:
            self.sign_frequency[client_id] = rising
            self.updates_seen[client_id] = 1
            return {key: torch.ones_like(value) for key, value in update.items()}
        frequency = self.sign_frequency[client_id]
        mask = {
            key: (
                torch.where(update[key] >= 0, frequency[key], 1.0 - frequency[key])
                > self.config.consistency_threshold
            ).float()
            for key in update
        }
        self.sign_frequency[client_id] = {
            key: (frequency[key] * seen + rising[key]) / (seen + 1) for key in update
        }
        self.updates_seen[client_id] = seen + 1
        return mask

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        if not uplinks:
            raise ValueError("aggregation requires nonempty client uplinks")
        if self.initial is None:
            # Both banks start from the initial global model (Algorithm 1 line 2).
            self.initial = self._flat(state=global_params)
            self.global_bank = [self.initial]
        clients = [int(message.sender.split(":")[1]) for message in uplinks]
        self.client_ids.update(clients)
        updates = {
            client_id: {
                key: message.tensors[key] - global_params[key] for key in global_params
            }
            for client_id, message in zip(clients, uplinks, strict=True)
        }
        code = self.config.reference == "code"
        if code:
            # paper vs code: the code masks the updates it aggregates, with the
            # running sign frequency of each client (the paper masks only the D_ct
            # distance loss), and averages them unweighted.
            masks = {
                client_id: (
                    self.sign_mask(client_id=client_id, update=update)
                    if self.config.consistency
                    else {key: torch.ones_like(value) for key, value in update.items()}
                )
                for client_id, update in updates.items()
            }
            masked = {
                client_id: {
                    key: masks[client_id][key] * value for key, value in update.items()
                }
                for client_id, update in updates.items()
            }
            aggregated = {
                key: global_params[key]
                + sum(update[key] for update in masked.values()) / len(masked)
                for key in global_params
            }
            # official code: the D_ct trajectories hold the masked client models.
            uploaded = {
                client_id: {key: global_params[key] + update[key] for key in update}
                for client_id, update in masked.items()
            }
        else:
            aggregated = Traditional.aggregate(
                self, uplinks=uplinks, global_params=global_params
            )
            uploaded = {
                client_id: dict(message.tensors)
                for client_id, message in zip(clients, uplinks, strict=True)
            }
        if self.config.client_synthetic:
            self._client_trajectories(updates=updates, uploaded=uploaded, code=code)
        if self.config.global_refinement:
            aggregated = self._global_trajectory(aggregated=aggregated, code=code)
        self.current_iteration += 1
        return aggregated

    def _client_trajectories(
        self,
        *,
        updates: dict[int, dict[str, Tensor]],
        uploaded: dict[int, dict[str, Tensor]],
        code: bool,
    ) -> None:
        ends = {
            client_id: self._flat(state=state) for client_id, state in uploaded.items()
        }
        assert self.initial is not None
        for client_id in ends:
            self.client_start.setdefault(client_id, self.initial)
        due = self.current_iteration % self.config.client_interval == 0
        if due:
            masks = {}
            for client_id in ends:
                previous = self.previous_update.get(client_id)
                if code or not self.config.consistency or previous is None:
                    masks[client_id] = torch.ones_like(ends[client_id])
                else:
                    # Paper: an element counts in the distance loss only when its
                    # update keeps the sign of the previous round's update.
                    current = self._flat(state=updates[client_id])
                    masks[client_id] = (
                        torch.sign(current) == torch.sign(self._flat(state=previous))
                    ).float()
            segments = {
                client_id: (
                    self.client_start[client_id],
                    ends[client_id],
                    masks[client_id],
                )
                for client_id in ends
            }
            order = sorted(segments)
            generator = self._generator(operation="fedtrend-client")

            def segment() -> tuple[Tensor, Tensor, Tensor]:
                index = int(torch.randint(len(order), (1,), generator=generator))
                return segments[order[index]]

            self.client_synthetic = self.construct(
                size=self.config.client_synthetic_size,
                segment=segment,
                distance=self.client_distance,
                generator=generator,
            )
            self.client_version += 1
        for client_id, end in ends.items():
            # paper vs code: the code's segment starts at the client's previous-round
            # model (one round long); the paper's at its model from the previous
            # construction (L_ct rounds long).
            if code or due:
                self.client_start[client_id] = end
        self.previous_update.update(updates)

    def _global_trajectory(
        self, *, aggregated: dict[str, Tensor], code: bool
    ) -> dict[str, Tensor]:
        if (
            self.current_iteration % self.config.global_interval == 0
            and self.current_iteration != 0
        ):
            bank = self.global_bank
            generator = self._generator(operation="fedtrend-global")
            segment_length = self.config.global_segment
            # paper vs code: the code draws the start below interval - segment; the
            # paper uniformly over the stored trajectory.
            high = (
                self.config.global_interval - segment_length
                if code
                else len(bank) - segment_length
            )
            ones = torch.ones_like(bank[0])

            def segment() -> tuple[Tensor, Tensor, Tensor]:
                start = int(torch.randint(high, (1,), generator=generator))
                return bank[start], bank[start + segment_length], ones

            self.global_synthetic = self.construct(
                size=self.config.global_synthetic_size or len(self.client_ids),
                segment=segment,
                distance=self.global_distance,
                generator=generator,
            )
            self.refined = 0
            # The bank restarts from the model of the construction round.
            self.global_bank = [self._flat(state=aggregated)]
        else:
            self.global_bank.append(self._flat(state=aggregated))
        if self.global_synthetic is not None and (
            not code or self.refined < self.config.refinement_rounds
        ):
            aggregated = self.refine(state=aggregated)
            self.refined += 1
        return aggregated
