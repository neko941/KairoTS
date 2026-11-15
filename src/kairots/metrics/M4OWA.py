"""M4 Overall Weighted Average against the pinned Naive2 competition baseline."""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class M4Context:
    baseline_smape: float
    baseline_mase: float


# From the released Naive2 forecasts at the pinned M4-methods revision, rescored
# on its pinned folds and checked against that revision's Evaluation and Ranks.xlsx.
_BASELINES = {
    "M4Yearly": (16.342173488085, 3.974360265174, 1, 6, 23000),
    "M4Quarterly": (11.011569928825, 1.371382270977, 4, 8, 24000),
    "M4Monthly": (14.426828759880, 1.063359447971, 12, 18, 48000),
    "M4Weekly": (9.161286913982, 2.777294993197, 1, 13, 359),
    "M4Daily": (3.045251780079, 3.278424295960, 1, 14, 4227),
    "M4Hourly": (18.382878117866, 2.395040006949, 24, 48, 414),
}


def overall_owa(*, fold_smape: dict[str, float], fold_mase: dict[str, float]) -> float:
    """Combine the six per-series fold means into the official M4 total OWA."""
    if set(fold_smape) != set(_BASELINES) or set(fold_mase) != set(_BASELINES):
        raise ValueError("overall M4 OWA requires scores for all six frequency folds")
    if any(
        not math.isfinite(score)
        for scores in (fold_smape, fold_mase)
        for score in scores.values()
    ):
        raise ValueError("overall M4 OWA requires finite fold scores")
    series_count = sum(baseline[4] for baseline in _BASELINES.values())
    model_smape = (
        sum(_BASELINES[name][4] * fold_smape[name] for name in _BASELINES)
        / series_count
    )
    model_mase = (
        sum(_BASELINES[name][4] * fold_mase[name] for name in _BASELINES) / series_count
    )
    baseline_smape = (
        sum(_BASELINES[name][4] * _BASELINES[name][0] for name in _BASELINES)
        / series_count
    )
    baseline_mase = (
        sum(_BASELINES[name][4] * _BASELINES[name][1] for name in _BASELINES)
        / series_count
    )
    return 0.5 * (model_smape / baseline_smape + model_mase / baseline_mase)


class M4OWA:
    CONTEXT = "m4_context"
    ROLES = ("metric",)
    TASKS = ("Forecasting",)
    NEEDS = ("point_prediction", "target", "seasonal_scale", "m4_context")
    PHASES = ("test",)
    DIRECTION = "min"

    @dataclass(kw_only=True)
    class Config:
        name: str = "M4OWA"

    def __init__(self, *, config: Config | None = None) -> None:
        self.smape_total: Tensor | None = None
        self.mase_total: Tensor | None = None
        self.count: Tensor | None = None
        self.context: M4Context | None = None

    @classmethod
    def prepare_context(cls, *, experiment: Any) -> M4Context:
        source = experiment.data.source
        baseline = _BASELINES.get(source.name)
        if baseline is None:
            raise ValueError("M4OWA requires an official M4 frequency source")
        baseline_smape, baseline_mase, seasonality, output_length, _ = baseline
        if source.seasonality != seasonality:
            raise ValueError("M4OWA source seasonality differs from its reference")
        if experiment.data.split.output_length != output_length:
            raise ValueError("M4OWA forecast horizon differs from its reference")
        return M4Context(baseline_smape=baseline_smape, baseline_mase=baseline_mase)

    def update(
        self,
        *,
        prediction: Tensor,
        target: Tensor,
        seasonal_scale: Tensor,
        m4_context: M4Context,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError(
                "M4OWA expects matching [batch, horizon, channels] tensors"
            )
        if seasonal_scale.shape != (prediction.shape[0], prediction.shape[2]):
            raise ValueError("M4OWA seasonal_scale must have shape [batch, channels]")
        if self.context is not None and self.context != m4_context:
            raise ValueError("M4OWA cannot merge different frequency references")
        self.context = m4_context

        scored = target.isfinite()
        prediction = prediction.double()
        target = target.double()
        denominator = prediction.abs() + target.abs()
        if torch.any(scored & (denominator == 0)):
            raise ValueError("M4OWA sMAPE is undefined when both values are zero")
        smape = torch.where(
            scored,
            200 * (prediction - target).abs() / denominator,
            0.0,
        ).sum()
        scale = seasonal_scale.double().to(device=prediction.device)
        if torch.any(scale == 0):
            raise ValueError("M4OWA MASE is undefined for a zero seasonal scale")
        mase = torch.where(
            scored,
            (prediction - target).abs() / scale[:, None, :],
            0.0,
        ).sum()
        count = scored.sum()
        self.smape_total = (
            smape if self.smape_total is None else self.smape_total + smape
        )
        self.mase_total = mase if self.mase_total is None else self.mase_total + mase
        self.count = count if self.count is None else self.count + count

    def merge(self, *, other: M4OWA) -> None:
        if other.count is None:
            return
        if other.smape_total is None or other.mase_total is None:
            raise ValueError("M4OWA accumulator is incomplete")
        if self.context is not None and self.context != other.context:
            raise ValueError("M4OWA cannot merge different frequency references")
        self.context = other.context
        if self.smape_total is None:
            self.smape_total = other.smape_total
        else:
            self.smape_total += other.smape_total
        if self.mase_total is None:
            self.mase_total = other.mase_total
        else:
            self.mase_total += other.mase_total
        self.count = other.count if self.count is None else self.count + other.count

    def compute(self) -> float:
        if (
            self.context is None
            or self.count is None
            or not self.count
            or self.smape_total is None
            or self.mase_total is None
        ):
            raise ValueError("M4OWA has no observations")
        smape = self.smape_total / self.count
        mase = self.mase_total / self.count
        return float(
            (
                0.5
                * (
                    smape / self.context.baseline_smape
                    + mase / self.context.baseline_mase
                )
            ).item()
        )
