"""Difference in Hill extreme-value indices of real and generated magnitudes.

Hill (1975), as stated in Kim and Lee (Bernoulli 2009), Eq. (2.4), estimates
xi = 1/alpha by mean log excess over the k+1 largest value. This metric compares
absolute returns, per channel, with a fixed tail count; it does not estimate an
optimal threshold or test whether a distribution has a Pareto tail.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.metrics.VaRError import VaRError


def hill(*, values: Tensor, tail_count: int) -> Tensor:
    """Hill xi for non-negative (observations, channels) and explicit tail count k."""
    if values.ndim != 2 or not 0 < tail_count < values.shape[0]:
        raise ValueError("Hill needs a matrix and 0 < tail_count < observations")
    if not values.isfinite().all() or (values < 0).any():
        raise ValueError("Hill needs finite non-negative observations")
    largest = values.double().topk(tail_count + 1, dim=0).values
    if (largest[-1] <= 0).any():
        raise ValueError("Hill needs a positive tail threshold in every channel")
    return (largest[:-1].log() - largest[-1].log()).mean(dim=0)


class HillError(VaRError):
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config(VaRError.Config):
        name: str = "HillError"
        tail_count: int = 50

        def __post_init__(self) -> None:
            if self.tail_count < 1:
                raise ValueError("HillError tail_count must be positive")

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config = config or self.Config()

    def compute(self) -> float:
        if not self.real:
            raise ValueError("HillError has no observations")
        real, generated = (
            hill(values=torch.cat(values).abs(), tail_count=self.config.tail_count)
            for values in (self.real, self.generated)
        )
        return float((real - generated).abs().mean())
