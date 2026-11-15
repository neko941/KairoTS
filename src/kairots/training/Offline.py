"""Ordinary offline training: every update sees the full training split."""

from dataclasses import dataclass
from typing import Any

from kairots.runtime.contracts import ClientState


class Offline:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Offline"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def select(
        self,
        *,
        windows: list[Any],
        current_iteration: int,
        iterations: int,
        num_classes: int | None,
        state: ClientState,
        seed: int,
    ) -> list[Any]:
        return windows

    def commit(self, *, state: ClientState) -> None:
        """Offline training keeps no release or replay state."""
