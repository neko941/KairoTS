"""Stable seed derivation from logical experiment identities."""

import hashlib
import os

import torch


def derive_seed(
    *, seed: int, current_iteration: int, client_id: int, operation: str
) -> int:
    payload = f"{seed}:{current_iteration}:{client_id}:{operation}".encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


def configure_determinism(*, deterministic: bool) -> None:
    """Make torch kernels reproducible; call in every process before CUDA work."""
    if deterministic:
        # cuBLAS needs a fixed workspace for deterministic matmuls (CUDA >= 10.2).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(mode=deterministic)
    torch.backends.cudnn.deterministic = deterministic
