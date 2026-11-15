"""Mergeable cross-channel correlation error: the Frobenius norm of the difference
between the real and generated channel correlation matrices, each over every step of
every window. A constant channel correlates 0 with the others. Merging adds the
sufficient statistics (row count, value sums, cross products).
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def correlation(*, count: int, total: Tensor, products: Tensor) -> Tensor:
    """Channel correlation matrix from the row count, value sums and cross products."""
    mean = total / count
    covariance = products / count - mean[:, None] * mean[None, :]
    deviation = covariance.diagonal().clamp_min(1e-12).sqrt()
    return covariance / (deviation[:, None] * deviation[None, :])


@dataclass(kw_only=True)
class CorrelationError:
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target")
    PHASES = ("test",)
    DIRECTION = "min"

    count: int = 0
    # Value sums and cross products, real then generated.
    totals: tuple[Tensor, Tensor] | None = None
    products: tuple[Tensor, Tensor] | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "CorrelationError"

    def __init__(self, *, config: Config | None = None) -> None:
        self.count = 0
        self.totals = None
        self.products = None

    def update(self, *, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("CorrelationError needs as many generated as real windows")
        real = target.flatten(0, 1).double()
        generated = prediction.flatten(0, 1).double()
        self._add(
            count=real.shape[0],
            totals=(real.sum(dim=0), generated.sum(dim=0)),
            products=(real.T @ real, generated.T @ generated),
        )

    def _add(
        self,
        *,
        count: int,
        totals: tuple[Tensor, Tensor],
        products: tuple[Tensor, Tensor],
    ) -> None:
        self.count += count
        if self.totals is None or self.products is None:
            self.totals, self.products = totals, products
            return
        self.totals = (self.totals[0] + totals[0], self.totals[1] + totals[1])
        self.products = (
            self.products[0] + products[0],
            self.products[1] + products[1],
        )

    def merge(self, *, other: CorrelationError) -> None:
        if other.totals is not None and other.products is not None:
            self._add(count=other.count, totals=other.totals, products=other.products)

    def compute(self) -> float:
        if self.totals is None or self.products is None or not self.count:
            raise ValueError("CorrelationError has no observations")
        real, generated = (
            correlation(count=self.count, total=total, products=products)
            for total, products in zip(self.totals, self.products, strict=True)
        )
        return float(torch.linalg.matrix_norm(real - generated))
