"""Held-out clients: windows split as `CompleteWindowRatio`, and a seeded share of
clients never trains; they are scored on test as the `unseen` population (scaled on
their own early portion, as TSLib scales a cross-dataset target)."""

import random
from dataclasses import dataclass

from kairots.data.splits.CompleteWindowRatio import CompleteWindowRatio


class Grouped(CompleteWindowRatio):
    @dataclass(kw_only=True)
    class Config(CompleteWindowRatio.Config):
        name: str = "Grouped"
        unseen_ratio: float = 0.25
        unseen_seed: int = 0

    def unseen(self, *, client_ids: list[int]) -> list[int]:
        """At least one unseen client when the ratio is positive; one stays seen."""
        config = self.config
        assert isinstance(config, Grouped.Config)
        if config.unseen_ratio <= 0 or len(client_ids) < 2:
            return []
        count = min(
            len(client_ids) - 1,
            max(1, int(len(client_ids) * config.unseen_ratio + 0.5)),
        )
        return sorted(random.Random(config.unseen_seed).sample(client_ids, count))
