"""Initialize selected linear weights to their input average, preserving biases."""

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

from torch import nn

from kairots.initializers.base import Policy


class Average(Policy):
    @dataclass(kw_only=True)
    class Config(Policy.Config):
        name: str = "Average"
        patterns: tuple[str, ...] = ("*",)

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def apply(
        self, *, model: nn.Module, manifest: dict[str, Any] | None = None
    ) -> dict[str, str]:
        weights = {
            id(module.weight)
            for module in model.modules()
            if isinstance(module, nn.Linear)
        }
        parameters = {
            name for name, value in model.named_parameters() if id(value) in weights
        }
        rules: list[tuple[str, str, dict[str, Any]]] = []
        for pattern in self.config.patterns:
            matched = sorted(name for name in parameters if fnmatch(name, pattern))
            if not matched:
                raise ValueError(f"Average: rule {pattern!r} matches no linear weight")
            rules.extend((name, "inverse_fan_in", {}) for name in matched)
        if not rules:
            raise ValueError("Average requires at least one parameter pattern")
        self.RULES = tuple(rules)
        return super().apply(model=model, manifest=manifest)
