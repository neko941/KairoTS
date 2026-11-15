"""Imputation: reconstruct values hidden from a window, scored only where hidden.

The split gives target windows (`input_length = 0`, `output_length = L`).
`prepare` hides a random `mask_ratio` of the observed positions:

- model input: the window with hidden and natively missing positions set to 0;
- target: the true value at hidden positions and NaN elsewhere, so losses and
  metrics score hidden positions only (a non-finite target is unscored).

Training masks continue the job's generator (fresh each batch); scoring masks restart
from `seed` for every client and evaluation, so validation and test always hide the
same positions.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class Imputation:
    HEAD = "Forecasting"
    RECONSTRUCTS = True
    LOSS = "MSE"
    METRICS = ("MAE",)

    @dataclass(kw_only=True)
    class Config:
        name: str = "Imputation"
        mask_ratio: float = 0.25
        seed: int = 0

        def __post_init__(self) -> None:
            if not 0.0 < self.mask_ratio < 1.0:
                raise ValueError("Imputation mask_ratio must be in (0, 1)")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def prepare(
        self,
        *,
        batch_x: Tensor,
        batch_y: Tensor,
        output_length: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if generator is None:
            raise ValueError("Imputation needs a generator to draw masks")
        observed = batch_y.isfinite()
        # Masks are drawn per batch in window order, so a fixed seed fixes
        # them only for a given batch size; key them on window positions if scores
        # must not depend on batch_size.
        hidden = (
            torch.rand(batch_y.shape, generator=generator) < self.config.mask_ratio
        ) & observed
        if not hidden.any():
            # A draw that hides nothing hides one observed position instead.
            positions = observed.flatten().nonzero().flatten()
            if not len(positions):
                raise ValueError("Imputation batch has no observed positions")
            choice = torch.randint(len(positions), (1,), generator=generator)
            hidden = hidden.flatten()
            hidden[positions[choice]] = True
            hidden = hidden.view(batch_y.shape)
        model_input = torch.where(hidden | ~observed, 0.0, batch_y)
        target = torch.where(hidden, batch_y, torch.nan)
        return model_input, target
