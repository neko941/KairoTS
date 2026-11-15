"""Online regime: supervised updates on a time-ordered stream with delayed targets.

A window's targets are released once the stream has observed its last target step.
Before a batch is predicted, every released window not yet learned from is trained
on, as one batch, with `steps` steps of the experiment learner's optimizer and the
evaluation loss; the predicted batch then waits until its own targets are due. No
update uses a target the stream has not observed at the forecast origin.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.optimizers.base import UpdateBudget
from kairots.registry import get
from kairots.replay.base import Replay


@dataclass(kw_only=True)
class Online:
    REQUIRES = {"training.name": (None, "Offline")}

    @dataclass(kw_only=True)
    class Config(Replay.Config, UpdateBudget.Config):
        name: str = "Online"

        def __post_init__(self) -> None:
            if self.steps < 0:
                raise ValueError("steps must be non-negative")
            if self.replay_size < 0 or self.capacity < 1:
                raise ValueError(
                    "replay_size must be non-negative and capacity positive"
                )

    def __init__(self, *, config: Config) -> None:
        self.config = config
        self.pending: list[tuple[Tensor, Tensor, int]] = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.updates = 0
        self.replay = None
        if config.replay_size:
            replay_type = get(category="replay", name=config.replay)
            self.replay = replay_type(
                config=replay_type.Config(capacity=config.capacity)
            )

    def start(self, *, optimizer: torch.optim.Optimizer) -> None:
        """Begin one client's stream: no pending targets, a fresh optimizer."""
        self.pending = []
        self.optimizer = optimizer
        if self.replay is not None:
            self.replay.reset()

    def observe(self, *, batch_x: Tensor, batch_y: Tensor, ends: list[int]) -> None:
        """Queue predicted windows; `ends` are their last target step plus one."""
        self.pending.extend(zip(batch_x, batch_y, ends, strict=True))

    def due(self, *, observed: int) -> list[tuple[Tensor, Tensor, int]]:
        """Remove and return queued windows whose targets end by `observed` steps."""
        due = [item for item in self.pending if item[2] <= observed]
        self.pending = [item for item in self.pending if item[2] > observed]
        return due

    def fit(
        self,
        *,
        model: nn.Module,
        items: list[tuple[Tensor, Tensor, int]],
        loss_function: Any,
    ) -> None:
        """`steps` optimizer steps on `items` as one batch."""
        if not items or self.config.steps == 0:
            return
        assert self.optimizer is not None
        batch_x = torch.stack([item[0] for item in items])
        batch_y = torch.stack([item[1] for item in items])
        model.train()
        with torch.enable_grad():
            for _ in range(self.config.steps):
                self.optimizer.zero_grad()
                loss_function(
                    prediction=model(batch_x=batch_x), target=batch_y
                ).backward()
                self.optimizer.step()
                self.updates += 1
        model.eval()

    def release(self, *, model: nn.Module, observed: int, loss_function: Any) -> None:
        """Train on queued windows whose targets end at or before `observed` steps."""
        due = self.due(observed=observed)
        if not due:
            return
        replayed = (
            []
            if self.replay is None
            else self.replay.sample(count=self.config.replay_size)
        )
        self.fit(model=model, items=due + replayed, loss_function=loss_function)
        if self.replay is not None:
            self.replay.admit(items=due)
