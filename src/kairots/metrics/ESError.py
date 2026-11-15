"""Mergeable tail-risk check: `VaRError` for the Expected Shortfall, the mean of the
values at or below the `LEVEL` quantile, per channel. Merging keeps the values.
"""

from dataclasses import dataclass

from kairots.metrics.VaRError import VaRError


class ESError(VaRError):
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config(VaRError.Config):
        name: str = "ESError"

    def compute(self) -> float:
        (_, real), (_, generated) = self._risks()
        return float((real - generated).abs().mean())
