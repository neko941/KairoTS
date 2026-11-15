"""Choose a worker/evaluation device from the resolved executor policy."""

import torch


def execution_device(*, policy: str, remote: bool, num_gpus: float) -> torch.device:
    if policy == "cpu":
        return torch.device("cpu")
    allocated = not remote or num_gpus > 0
    if policy == "cuda" or allocated:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if policy == "cuda" or remote:
            raise RuntimeError("CUDA was requested but is unavailable in this worker")
    return torch.device("cpu")
