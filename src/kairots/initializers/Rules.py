"""Apply configured parameter-pattern initialization rules in order."""

from dataclasses import dataclass
from typing import Any

from kairots.initializers.base import INITIALIZERS, Policy


class Rules(Policy):
    @dataclass(kw_only=True)
    class Config(Policy.Config):
        name: str = "Rules"
        rules: tuple[tuple[str, str, dict[str, Any]], ...] = ()

        def __post_init__(self) -> None:
            for pattern, initializer, options in self.rules:
                if not isinstance(pattern, str) or not isinstance(options, dict):
                    raise TypeError("Rules need a pattern string and options object")
                if initializer not in INITIALIZERS:
                    raise ValueError(f"unknown initializer {initializer!r}")

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.RULES = tuple(config.rules)
