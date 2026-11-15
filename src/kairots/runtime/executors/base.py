"""Options shared by local and remote executor profiles."""

import math
from dataclasses import dataclass


class Executor:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Executor"
        device: str = "auto"
        num_workers: int = 2
        num_gpus: float = 0.0
        deterministic: bool = True
        seed: int = 941
        runs_dir: str = "runs"
        save_checkpoints: bool = True
        # Resume from this run's checkpoints/resume.pt.
        resume: str = ""

        def __post_init__(self) -> None:
            if not isinstance(self.device, str) or self.device not in (
                "auto",
                "cpu",
                "cuda",
            ):
                raise ValueError("runtime.device must be 'auto', 'cpu', or 'cuda'")
            if (
                isinstance(self.num_workers, bool)
                or not isinstance(self.num_workers, int)
                or self.num_workers < 1
            ):
                raise ValueError("runtime.num_workers must be >= 1")
            if (
                isinstance(self.num_gpus, bool)
                or not isinstance(self.num_gpus, (int, float))
                or not math.isfinite(self.num_gpus)
                or self.num_gpus < 0
            ):
                raise ValueError("runtime.num_gpus must be finite and >= 0")
