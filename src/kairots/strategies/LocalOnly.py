"""Local-only baseline: each client trains alone, nothing is aggregated."""

from dataclasses import dataclass

from kairots.strategies.NonFederated import NonFederated


class LocalOnly(NonFederated):
    @dataclass(kw_only=True)
    class Config(NonFederated.Config):
        name: str = "LocalOnly"
