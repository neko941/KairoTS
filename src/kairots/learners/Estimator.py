"""Native fit for estimators with their own training (XGBoost): no optimizer; each
epoch is one `model.fit` call on every training window."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.learners.base import Learner


@dataclass(kw_only=True)
class Estimator:
    @dataclass(kw_only=True)
    class Config(Learner.Config):
        name: str = "Estimator"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def optimizer(self, *, model: nn.Module) -> None:
        return None

    def fit_epoch(
        self,
        *,
        model: Any,
        optimizer: Any,
        batches: list[tuple[Tensor, Tensor]],
        device: torch.device,
        loss_function: Any,
        regularizer: Any,
    ) -> float:
        """Fit on all windows without the gradient learner's update machinery.

        An estimator owns its objective, so `loss_function` and `regularizer` do not
        define its training loss.
        """
        return model.fit(
            batch_x=torch.cat([batch_x for batch_x, _ in batches]).cpu(),
            batch_y=torch.cat([batch_y for _, batch_y in batches]).cpu(),
        )
