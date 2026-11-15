"""Persistent Ray actor pool with one object reference per client dataset."""

import os
from dataclasses import dataclass
from typing import Any

import ray  # type: ignore[import-not-found]
from torch import Tensor

from kairots.config.schema import Experiment
from kairots.data.contracts import ClientRecord
from kairots.runtime.contracts import Job, Message, Result
from kairots.runtime.executors.base import Executor
from kairots.runtime.worker import Worker


@dataclass(kw_only=True)
class Ray:
    REMOTE = True
    experiment: Experiment
    actors: list[Any]
    owners: dict[int, int]
    initialized_here: bool

    @dataclass(kw_only=True)
    class Config(Executor.Config):
        name: str = "Ray"

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.device == "cuda" and self.num_gpus == 0:
                raise ValueError("Ray with runtime.device='cuda' needs num_gpus > 0")
            if self.device == "cpu" and self.num_gpus > 0:
                raise ValueError("Ray with runtime.device='cpu' needs num_gpus=0")

    def __init__(self, *, experiment: Experiment, worker: Worker) -> None:
        self.experiment = experiment
        self.initialized_here = not ray.is_initialized()
        if self.initialized_here:
            os.environ.setdefault("RAY_AUTH_MODE", "disabled")
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                num_cpus=max(experiment.runtime.num_workers, 1),
            )
        remote_worker = ray.remote(num_gpus=experiment.runtime.num_gpus)(Worker)
        self.actors = [
            remote_worker.remote(
                experiment=worker.experiment,
                strategy=worker.strategy,
                task=worker.task,
                learner_type=worker.learner_type,
                loss_function=worker.loss_function,
                view=worker.view,
            )
            for _ in range(experiment.runtime.num_workers)
        ]
        self.owners: dict[int, int] = {}

    def cache_client(self, *, client_id: int, data: ClientRecord) -> None:
        # Every actor knows every record (paths only; values load on first use), so a
        # pooled job (Centralized) can read all clients wherever it runs.
        self.owners[client_id] = client_id % len(self.actors)
        record = ray.put(data)
        for actor in self.actors:
            actor.cache_client.remote(client_id=client_id, data=record)

    def execute(self, *, jobs: list[Job], broadcast: dict[str, Tensor]) -> list[Result]:
        broadcast_ref = ray.put(broadcast)
        futures = []
        for job in jobs:
            owner = self.owners[job.client_id]
            light_job = Job(
                operation_id=job.operation_id,
                operation=job.operation,
                # Only tensors that are not the broadcast ones (per-client extras)
                # travel with the job.
                downlink=Message(
                    sender=job.downlink.sender,
                    receiver=job.downlink.receiver,
                    tensors={
                        key: value
                        for key, value in job.downlink.tensors.items()
                        if broadcast.get(key) is not value
                    },
                ),
                state=job.state,
                seed=job.seed,
                client_id=job.client_id,
            )
            futures.append(
                self.actors[owner].execute.remote(
                    job=light_job,
                    # An empty downlink (local-only) stays empty.
                    broadcast=broadcast_ref if job.downlink.tensors else None,
                )
            )
        return list(ray.get(futures))

    def close(self) -> None:
        for actor in self.actors:
            ray.kill(actor)
        if self.initialized_here:
            ray.shutdown()
