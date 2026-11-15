"""Reconstruction-based anomaly detection (TSLib `exp_anomaly_detection`).

Windows are target windows (`input_length = 0`, `output_length = L`) of a series whose
last channel is the per-step label. The model reconstructs the window; training is
normal-only, so labels are never read here. Scoring and thresholds belong to the
`Calibrated` evaluation protocol.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class AnomalyDetection:
    HEAD = "Forecasting"
    RECONSTRUCTS = True
    LOSS = "MSE"
    METRICS = ("F1", "AdjustedF1")
    PROTOCOL = "Calibrated"

    @dataclass(kw_only=True)
    class Config:
        name: str = "AnomalyDetection"

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
        values = batch_y[..., :-1]
        return values, values
