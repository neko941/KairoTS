"""Shared period releases and checkpointed replay for continual schedules."""

from dataclasses import dataclass
from typing import Any

from kairots.registry import get
from kairots.replay.base import Replay
from kairots.runtime.contracts import ClientState
from kairots.training.Offline import Offline


class Continual(Offline):
    ABSTRACT = True

    @dataclass(kw_only=True)
    class Config(Offline.Config, Replay.Config):
        name: str = "Continual"
        periods: int = 2

        def __post_init__(self) -> None:
            if self.periods < 2 or self.capacity < 1 or self.replay_size < 0:
                raise ValueError(
                    "Continual needs periods >= 2, capacity > 0 and replay_size >= 0"
                )

    def __init__(self, *, config: Config) -> None:
        self.config = config
        self.due: list[Any] = []
        self.replay: Any = None

    def windows(
        self,
        *,
        windows: list[tuple[int, int]],
        period: int,
        num_classes: int | None,
    ) -> list[tuple[int, int]]:
        """Select a period's windows in the concrete schedule."""
        raise NotImplementedError

    def released(
        self,
        *,
        windows: list[tuple[int, int]],
        current_iteration: int,
        iterations: int,
        num_classes: int | None,
    ) -> list[tuple[int, int]]:
        """A disjoint share of the current period's examples for this round."""
        return self.windows(
            windows=windows,
            period=current_iteration // iterations,
            num_classes=num_classes,
        )[current_iteration % iterations :: iterations]

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
        self.due = self.released(
            windows=windows,
            current_iteration=current_iteration,
            iterations=iterations,
            num_classes=num_classes,
        )
        replayed = []
        if self.config.replay_size:
            replay_type = get(category="replay", name=self.config.replay)
            self.replay = state.extra.get("replay")
            if self.replay is None:
                self.replay = replay_type(
                    config=replay_type.Config(
                        capacity=self.config.capacity,
                        seed=seed,
                    )
                )
            replayed = self.replay.sample(count=self.config.replay_size)
        return self.due + replayed

    def commit(self, *, state: ClientState) -> None:
        if self.replay is not None:
            self.replay.admit(items=self.due)
            state.extra["replay"] = self.replay
