"""Quantile forecasting: the model predicts `num_quantiles` levels per position.

Levels are evenly spaced, `(k + 1) / (num_quantiles + 1)` for `k < num_quantiles`
(9 gives 0.1, ..., 0.9), and are read from the prediction's last axis by the losses
and metrics, so the shape is their single source. Windows are prepared as in
`Forecasting`.
"""

from dataclasses import dataclass

from kairots.tasks.Forecasting import Forecasting


@dataclass(kw_only=True)
class ProbabilisticForecasting(Forecasting):
    HEAD = "Quantile"
    RECONSTRUCTS = False
    LOSS = "Pinball"
    METRICS = ("CRPS", "CoverageError", "IntervalWidth")

    @dataclass(kw_only=True)
    class Config(Forecasting.Config):
        name: str = "ProbabilisticForecasting"
        num_quantiles: int = 9

        def __post_init__(self) -> None:
            if self.num_quantiles < 1:
                raise ValueError("ProbabilisticForecasting num_quantiles must be >= 1")

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
