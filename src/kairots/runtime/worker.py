"""Worker dispatches resolved strategy clients and component implementations."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
from torch import Tensor

from kairots.config.schema import Experiment
from kairots.data.contracts import ClientRecord
from kairots.registry import get, info
from kairots.runtime.contracts import ClientState, Job, Message, Result
from kairots.runtime.device import execution_device
from kairots.runtime.seeds import configure_determinism
from kairots.strategies.base import Strategy


@dataclass(kw_only=True)
class ClientData:
    values: Tensor
    # (start, target) rows for series; (sample, label) for panel data.
    splits: dict[str, list[tuple[int, int]]]


class Batches:
    """Training batches gathered again on every pass instead of held in memory.
    Each pass replays the job's generator from one saved state, so every pass
    (epoch) yields the same batches."""

    def __init__(
        self,
        *,
        stream: Callable[..., Iterator[tuple[Tensor, Tensor]]],
        state: Tensor,
        length: int,
    ) -> None:
        self.stream = stream
        self.state = state
        self.length = length
        self.sample_count = 0

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        generator = torch.Generator()
        generator.set_state(self.state)

        def batches() -> Iterator[tuple[Tensor, Tensor]]:
            self.sample_count = 0
            for batch in islice(self.stream(generator=generator), self.length):
                self.sample_count += len(batch[1])
                yield batch

        return batches()

    def __len__(self) -> int:
        return self.length


def _sample_windows(
    *,
    windows: list[tuple[int, int]],
    batch_size: int,
    batches_per_epoch: int,
    generator: torch.Generator,
) -> list[tuple[int, int]]:
    if not windows:
        raise ValueError("cannot sample training windows from an empty split")
    indices = torch.randint(
        len(windows), (batch_size * batches_per_epoch,), generator=generator
    ).tolist()
    return [windows[index] for index in indices]


class Worker:
    # shortcut: eight warm clients per worker; make this configurable if measured
    # workloads need a larger resident set.
    CLIENT_DATA_CACHE_LIMIT = 8

    def __init__(
        self,
        *,
        experiment: Experiment,
        strategy: Any,
        task: Any,
        learner_type: Any,
        loss_function: Any,
        view: Any,
    ) -> None:
        self.experiment = experiment
        self.strategy = strategy
        self.task = task
        self.learner_type = learner_type
        self.loss_function = loss_function
        self.view = view
        self.model_types: dict[str, Any] = {}
        for member in experiment.groups or (experiment,):
            name = member.model.name
            if name not in self.model_types:
                self.model_types[name] = get(category="models", name=name)
        self.augmentation = self._augmentation(experiment=experiment)
        # Each client group's task, view and augmentation (its own channels).
        self.members: dict[int, tuple[Any, Any, Any]] = {}
        self.records: dict[int, ClientRecord] = {}
        self.clients: OrderedDict[int, ClientData] = OrderedDict()
        configure_determinism(deterministic=experiment.runtime.deterministic)

    @staticmethod
    def _augmentation(*, experiment: Experiment) -> Any:
        augmentation = experiment.data.augmentation
        if augmentation is None:
            return None
        return get(category="augmentations", name=augmentation.name)(
            config=augmentation
        )

    def components(self, *, client_id: int) -> tuple[Any, Any, Any]:
        """(task, view, augmentation) of the client's group."""
        if not self.experiment.groups:
            return self.task, self.view, self.augmentation
        group = self.records[client_id].group
        if group not in self.members:
            experiment = self.experiment.member(group=group)
            self.members[group] = (
                get(category="tasks", name=experiment.task.name)(
                    config=experiment.task
                ),
                get(category="data/views", name=experiment.data.view.name)(
                    config=experiment.data.view
                ),
                self._augmentation(experiment=experiment),
            )
        return self.members[group]

    def cache_client(self, *, client_id: int, data: ClientRecord) -> None:
        self.records[client_id] = data

    def client_experiment(self, *, client_id: int) -> Experiment:
        """The experiment of the client's group."""
        return self.experiment.member(group=self.records[client_id].group)

    def client_data(self, *, client_id: int) -> ClientData:
        """Read from storage on a miss and retain only the eight most recent clients."""
        if client_id in self.clients:
            self.clients.move_to_end(client_id)
            return self.clients[client_id]
        record = self.records[client_id]
        store = get(category="data/stores", name=record.store)
        values, splits = store.load(record=record)
        self.clients[client_id] = ClientData(values=values, splits=splits)
        if len(self.clients) > self.CLIENT_DATA_CACHE_LIMIT:
            self.clients.popitem(last=False)
        return self.clients[client_id]

    @staticmethod
    def _cpu_state(*, value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: Worker._cpu_state(value=item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(Worker._cpu_state(value=item) for item in value)
        return value

    def execute(
        self, *, job: Job, broadcast: dict[str, Tensor] | None = None
    ) -> Result:
        started = time.perf_counter()
        if broadcast is not None:
            job.downlink.tensors = {**broadcast, **job.downlink.tensors}
        if job.operation == "finetune":
            # Every parameter trains, whatever the strategy freezes in its rounds.
            result = self.update(job=job, client=Strategy.Client())
        else:
            result = self.strategy.client_update(worker=self, job=job)
        result.timing["worker_seconds"] = time.perf_counter() - started
        return result

    def update(
        self, *, job: Job, client: Any, client_ids: list[int] | None = None
    ) -> Result:
        """Fit on `client_ids`' train windows (default: the job's client)."""
        client_ids = client_ids or [job.client_id]
        if job.operation not in ("fit", "finetune") or any(
            member_id not in self.records for member_id in client_ids
        ):
            raise ValueError(f"unsupported or uncached client job: {job.client_id}")
        experiment = self.client_experiment(client_id=job.client_id)
        if any(
            self.client_experiment(client_id=member_id).lengths()
            != experiment.lengths()
            for member_id in client_ids
        ):
            raise ValueError("pooled clients need the same window lengths")
        config = self.experiment.runtime
        device = execution_device(
            policy=config.device,
            remote=info(category="runtime/executors", name=config.name).get(
                "REMOTE", False
            ),
            num_gpus=config.num_gpus,
        )
        torch.manual_seed(job.seed)
        model = self.model_types[experiment.model.name](config=experiment.model).to(
            device
        )
        # The model a client starts the round from; proximal terms anchor on it.
        start = {**job.downlink.tensors, **job.state.model_state}
        model.load_state_dict(start)
        learner = self.learner_type(config=self.experiment.learner)
        optimizer = learner.optimizer(model=model)
        if job.state.optimizer_state:
            optimizer.load_state_dict(job.state.optimizer_state)
        input_length, output_length, offset_length = experiment.lengths()
        task, view, augmentation = self.components(client_id=job.client_id)
        batch_size = experiment.data.source.batch_size
        batches_per_epoch = experiment.learner.batches_per_epoch
        if batches_per_epoch is not None and len(client_ids) != 1:
            raise ValueError("learner.batches_per_epoch requires one training client")

        timing = {"data_seconds": 0.0}
        stream_timing = {"data_seconds": 0.0}
        data_started = time.perf_counter()
        train_windows = {
            member_id: self.client_data(client_id=member_id).splits["train"]
            for member_id in client_ids
        }
        training_type = get(
            category="training",
            name=experiment.training.name
            if experiment.training is not None
            else "Offline",
        )
        training = training_type(config=experiment.training or training_type.Config())
        for member_id, windows in train_windows.items():
            selected = training.select(
                windows=windows,
                current_iteration=job.operation_id[0],
                iterations=experiment.strategy.iterations,
                num_classes=getattr(experiment.data.source, "num_classes", None),
                state=job.state,
                seed=experiment.runtime.seed + job.client_id,
            )
            if not set(selected).issubset(windows):
                raise ValueError(
                    f"training schedule selected windows outside client "
                    f"{member_id}'s train split"
                )
            train_windows[member_id] = selected
        timing["data_seconds"] += time.perf_counter() - data_started

        def stream(*, generator: torch.Generator) -> Iterator[tuple[Tensor, Tensor]]:
            def client_batches() -> Iterator[tuple[Tensor, Tensor]]:
                for member_id in client_ids:
                    windows = train_windows[member_id]
                    if batches_per_epoch is not None:
                        windows = _sample_windows(
                            windows=windows,
                            batch_size=batch_size,
                            batches_per_epoch=batches_per_epoch,
                            generator=generator,
                        )
                    yield from view.batches(
                        values=self.client_data(client_id=member_id).values,
                        windows=windows,
                        input_length=input_length,
                        output_length=output_length,
                        offset_length=offset_length,
                        batch_size=batch_size,
                        shuffle=batches_per_epoch is None,
                        generator=generator,
                    )

            gathered: Any = client_batches()
            if len(client_ids) > 1:
                # Pooled training shuffles whole batches, each from one client, so
                # it holds one pass's batches; mix windows across clients if batch
                # composition matters.
                gathered = list(gathered)
                order = torch.randperm(len(gathered), generator=generator).tolist()
                gathered = [gathered[index] for index in order]
            for batch_x, batch_y in gathered:
                # Training inputs only, fresh noise per round; training masks
                # (imputation) continue the job's generator.
                if augmentation is not None:
                    batch_x = augmentation.apply(batch_x=batch_x, generator=generator)
                yield task.prepare(
                    batch_x=batch_x,
                    batch_y=batch_y,
                    output_length=output_length,
                    generator=generator,
                )

        def timed_stream(
            *, generator: torch.Generator
        ) -> Iterator[tuple[Tensor, Tensor]]:
            iterator = stream(generator=generator)
            while True:
                started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                stream_timing["data_seconds"] += time.perf_counter() - started
                yield batch

        batch_count = sum(
            math.ceil(len(train_windows[member_id]) / batch_size)
            for member_id in client_ids
        )
        if self.experiment.learner.batches_per_epoch is not None:
            batch_count = min(batch_count, self.experiment.learner.batches_per_epoch)
        batches = Batches(
            stream=timed_stream,
            state=torch.Generator().manual_seed(job.seed).get_state(),
            length=batch_count,
        )
        epochs = (
            getattr(self.experiment.evaluation, "finetune_epochs", 0)
            if job.operation == "finetune"
            else self.experiment.learner.epochs
        )
        # A run-span schedule steps once per client update and resumes from the
        # client's state; a round-span one restarts every update and steps after
        # every optimizer step of it.
        per_round = getattr(self.experiment.scheduler, "span", "run") == "round"
        if optimizer is not None and per_round:
            # A loaded optimizer carries the previous update's annealed rate.
            for group in optimizer.param_groups:
                group["lr"] = group.get("initial_lr", group["lr"])
        scheduler_type = get(category="schedulers", name=self.experiment.scheduler.name)
        step_per_batch = getattr(scheduler_type, "STEP_PER_BATCH", False)
        scheduler = (
            None
            if optimizer is None
            else scheduler_type(config=self.experiment.scheduler).build(
                optimizer=optimizer,
                iterations=(
                    self.experiment.strategy.iterations * epochs * len(batches)
                    if step_per_batch
                    else epochs * len(batches)
                    if per_round
                    else self.experiment.strategy.iterations
                ),
            )
        )
        if (
            optimizer is not None
            and scheduler is not None
            and (per_round or step_per_batch)
        ):
            optimizer.register_step_post_hook(lambda *_: scheduler.step())
        if (
            scheduler is not None
            and optimizer is not None
            and not per_round
            and job.state.scheduler_state
        ):
            scheduler.load_state_dict(job.state.scheduler_state)
            for group, rate in zip(
                optimizer.param_groups, scheduler.get_last_lr(), strict=True
            ):
                group["lr"] = rate
        fit_started = time.perf_counter()
        train_loss = client.fit(
            learner=learner,
            model=model,
            optimizer=optimizer,
            batches=batches,
            epochs=epochs,
            device=device,
            loss_function=self.loss_function,
            reference_parameters=start,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_seconds = time.perf_counter() - fit_started
        # One step per client update, so a client that skips rounds lags
        # the global schedule; step by current_iteration if that matters.
        if scheduler is not None and not per_round and not step_per_batch:
            scheduler.step()
        training.commit(state=job.state)
        uplink, kept = self.strategy.split_state(
            parameters={
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        )
        return Result(
            operation_id=job.operation_id,
            # The reply goes back the way the downlink came.
            uplink=Message(
                sender=job.downlink.receiver,
                receiver=job.downlink.sender,
                tensors=uplink,
                scalars=self.strategy.uplink_scalars(
                    tensors=uplink, train_samples=batches.sample_count
                ),
            ),
            state=ClientState(
                version=job.state.version + 1,
                optimizer_state=(
                    {}
                    if optimizer is None
                    else self._cpu_state(value=optimizer.state_dict())
                ),
                scheduler_state=(
                    {} if scheduler is None or per_round else scheduler.state_dict()
                ),
                model_state=kept,
                extra=job.state.extra,
            ),
            metrics={"train_loss": train_loss},
            train_samples=batches.sample_count,
            timing={
                "data_seconds": timing["data_seconds"] + stream_timing["data_seconds"],
                "fit_seconds": max(0.0, fit_seconds - stream_timing["data_seconds"]),
            },
        )
