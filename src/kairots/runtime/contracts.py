"""Shared client-server Message, Job and Result contract."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass(kw_only=True)
class Message:
    sender: str
    receiver: str
    tensors: dict[str, Tensor] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    # Size of the encoded payload when a codec compresses `tensors` (which then hold
    # the decoded values); None bills the raw tensors.
    encoded_bytes: int | None = None

    def __post_init__(self) -> None:
        """Everything a message carries is billed, so it carries only billable
        values: tensors, and numbers billed as 4-byte scalars."""
        for name, tensor in self.tensors.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"message tensor {name!r} is {type(tensor).__name__}")
        for name, scalar in self.scalars.items():
            if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
                raise TypeError(f"message scalar {name!r} is {type(scalar).__name__}")

    @property
    def billed_bytes(self) -> int:
        """Bytes on the simulated network; a message to oneself crosses none."""
        if self.sender == self.receiver:
            return 0
        if self.encoded_bytes is not None:
            return self.encoded_bytes + 4 * len(self.scalars)
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self.tensors.values()
        ) + 4 * len(self.scalars)


@dataclass(kw_only=True)
class ClientState:
    version: int = 0
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    scheduler_state: dict[str, Any] = field(default_factory=dict)
    # Parameters the client keeps for itself (personal or local-only models); they
    # override the downlink when training and evaluating, and are never billed.
    model_state: dict[str, Tensor] = field(default_factory=dict)
    # Method state the client carries between rounds (a personal model, control
    # variates); the worker passes it through unchanged unless the strategy sets it.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Job:
    operation_id: tuple[int, int, str]
    operation: str
    downlink: Message
    state: ClientState
    seed: int
    client_id: int


@dataclass(kw_only=True)
class Result:
    operation_id: tuple[int, int, str]
    uplink: Message
    state: ClientState
    metrics: dict[str, float]
    train_samples: int
    timing: dict[str, float] = field(default_factory=dict)
