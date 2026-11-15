"""F1 after point adjustment (TSLib `utils/tools.adjustment`).

If any step of a labeled anomaly segment is predicted, the whole segment counts as
detected. This inflates scores (Kim et al., AAAI 2022), so it is reported only under
its own name, beside the point-wise `F1`.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from kairots.metrics.F1 import F1


class AdjustedF1(F1):
    ROLES = ("metric",)
    TASKS = ("AnomalyDetection",)
    NEEDS = ("anomaly_predictions", "labels")
    DIRECTION = "max"
    PHASES = ("test",)

    @dataclass(kw_only=True)
    class Config(F1.Config):
        name: str = "AdjustedF1"

    @staticmethod
    def adjust(*, prediction: Tensor, target: Tensor) -> Tensor:
        """Mark every step of a labeled segment that has at least one hit."""
        starts = target & ~torch.cat([target.new_zeros(1), target[:-1]])
        segment = torch.cumsum(starts.long(), dim=0) * target
        hit = torch.zeros(int(segment.max()) + 1, dtype=torch.bool)
        hit[segment[prediction & target]] = True
        hit[0] = False
        return prediction | hit[segment]
