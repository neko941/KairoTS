"""Train on synthetic, test on real (Esteban et al., 2017), as a ratio to train on
real: a one-step linear autoregression on the previous `LAGS` steps of every channel
is fit by least squares once on the generated windows and once on the client's train
windows, and each predicts every step of the real held-out windows. The score is the
synthetic fit's MSE over the real fit's: 1 means the samples train as good a
predictor as the real data. Merging adds the squared errors.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def autoregression(*, windows: Tensor, lags: int) -> tuple[Tensor, Tensor]:
    """Features (the previous `lags` steps of every channel, plus a bias) and targets
    (the next step) for every step after the first `lags` of every window."""
    unfolded = windows.double().unfold(1, lags + 1, 1)
    features = unfolded[..., :-1].flatten(2).flatten(0, 1)
    bias = features.new_ones(features.shape[0], 1)
    return torch.cat([features, bias], dim=1), unfolded[..., -1].flatten(0, 1)


@dataclass(kw_only=True)
class TSTR:
    ROLES = ("metric",)
    TASKS = ("Generation",)
    NEEDS = ("samples", "target", "train")
    PHASES = ("test",)
    DIRECTION = "min"
    LAGS = 4

    synthetic_error: float = 0.0
    real_error: float = 0.0

    @dataclass(kw_only=True)
    class Config:
        name: str = "TSTR"

    def __init__(self, *, config: Config | None = None) -> None:
        self.synthetic_error = 0.0
        self.real_error = 0.0

    def update(self, *, prediction: Tensor, target: Tensor, train: Tensor) -> None:
        lags = min(self.LAGS, target.shape[1] - 1)
        if lags < 1:
            raise ValueError("TSTR needs windows of at least 2 steps")
        features, values = autoregression(windows=target, lags=lags)

        def error(*, windows: Tensor) -> float:
            fit_features, fit_values = autoregression(windows=windows, lags=lags)
            weights = torch.linalg.lstsq(fit_features, fit_values).solution
            return float((features @ weights - values).square().sum())

        self.synthetic_error += error(windows=prediction)
        self.real_error += error(windows=train)

    def merge(self, *, other: TSTR) -> None:
        self.synthetic_error += other.synthetic_error
        self.real_error += other.real_error

    def compute(self) -> float:
        if not self.real_error:
            raise ValueError("TSTR has no observations")
        return self.synthetic_error / self.real_error
