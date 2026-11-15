"""Mergeable mean absolute scaled error, TSLib `utils/m4_summary.py` `mase`:
`|prediction - target|` divided by the mean absolute seasonal difference
`|h[t] - h[t - m]|` of the window's history (m = seasonality), per channel,
pooled over scored positions; a zero scale counts as 1; a non-finite target marks
an unscored position. MASE is unchanged by a per-channel affine scaler, so scaled
values give the raw-scale result. `seasonal_scales` computes every window's scale
from one cumulative sum over the client's series.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def seasonal_scales(
    *, series: Tensor, target_rows: list[int], seasonality: int
) -> Tensor:
    """(windows, channels): the mean of `|h[t] - h[t - m]|` over the rows of
    `series` before each window's target."""
    if any(target <= seasonality for target in target_rows):
        raise ValueError("MASE history must be longer than seasonality")
    differences = (series[seasonality:] - series[:-seasonality]).abs().double()
    cumulative = torch.cat(
        (differences.new_zeros(1, differences.shape[1]), differences.cumsum(dim=0))
    )
    ends = torch.tensor(target_rows, device=series.device) - seasonality
    return (cumulative[ends] / ends[:, None]).to(dtype=series.dtype)


@dataclass(kw_only=True)
class MASE:
    ROLES = ("metric",)
    TASKS = ("Forecasting",)
    NEEDS = ("point_prediction", "target", "seasonal_scale")
    DIRECTION = "min"

    total: Tensor | None = None
    count: Tensor | None = None

    @dataclass(kw_only=True)
    class Config:
        name: str = "MASE"

    def __init__(self, *, config: Config | None = None) -> None:
        self.total = None
        self.count = None

    def update(
        self,
        *,
        prediction: Tensor,
        target: Tensor,
        seasonal_scale: Tensor,
    ) -> None:
        """`seasonal_scale` holds each window's scale, (windows, channels)."""
        scale = torch.where(seasonal_scale == 0, 1.0, seasonal_scale).to(
            device=prediction.device
        )
        scored = target.isfinite()
        error = torch.where(scored, (prediction - target).abs(), 0.0)
        value = (error / scale[:, None, :]).sum()
        count = scored.sum()
        self.total = value if self.total is None else self.total + value
        self.count = count if self.count is None else self.count + count

    def merge(self, *, other: MASE) -> None:
        if other.total is not None:
            self.total = other.total if self.total is None else self.total + other.total
        if other.count is not None:
            self.count = other.count if self.count is None else self.count + other.count

    def compute(self) -> float:
        if self.total is None or self.count is None or not self.count:
            raise ValueError("MASE has no observations")
        return float((self.total / self.count).item())
