"""Serial executor with one worker and a per-client data cache."""

from dataclasses import dataclass

from torch import Tensor

from kairots.config.schema import Experiment
from kairots.data.contracts import ClientRecord
from kairots.runtime.contracts import Job, Result
from kairots.runtime.executors.base import Executor
from kairots.runtime.worker import Worker


@dataclass(kw_only=True)
class Serial:
    @dataclass(kw_only=True)
    class Config(Executor.Config):
        name: str = "Serial"

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.num_gpus > 0:
                raise ValueError("runtime.num_gpus is only supported by Ray")

    def __init__(self, *, experiment: Experiment, worker: Worker) -> None:
        self.experiment = experiment
        self.worker = worker

    def cache_client(self, *, client_id: int, data: ClientRecord) -> None:
        self.worker.cache_client(client_id=client_id, data=data)

    def execute(self, *, jobs: list[Job], broadcast: dict[str, Tensor]) -> list[Result]:
        return [self.worker.execute(job=job) for job in jobs]

    def close(self) -> None:
        return None
