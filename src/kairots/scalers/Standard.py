"""Per-channel standard scaler, as scikit-learn's `StandardScaler` (TSLib).

A channel that is constant on the fitting rows keeps scale 1, so later changes in it
stay finite instead of being divided by ~0. Missing values require explicit masked
handling: statistics skip them and an unobserved channel keeps mean 0 and scale 1.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(kw_only=True)
class Standard:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Standard"
        imputation: str | None = None

        def __post_init__(self) -> None:
            if self.imputation not in (None, "masked"):
                raise ValueError("data.scaler.imputation must be None or 'masked'")

    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, *, values: Tensor, config: Config) -> Standard:
        if config.imputation is None and values.isnan().any():
            raise ValueError("missing values require data.scaler.imputation='masked'")
        if config.imputation == "masked" and values.isnan().any():
            mean = values.nanmean(dim=0)
            scale = (values - mean).square().nanmean(dim=0).sqrt().nan_to_num(nan=0.0)
            mean = mean.nan_to_num(nan=0.0)
        else:
            mean = values.mean(dim=0)
            scale = values.std(dim=0, unbiased=False)
        constant = scale < 10 * torch.finfo(scale.dtype).eps
        scale = torch.where(constant, torch.ones_like(scale), scale)
        return cls(mean=mean, scale=scale)

    @staticmethod
    def validate_values(
        *,
        values: Tensor,
        config: Config,
        source: str,
        client: str,
        channels: tuple[str, ...],
    ) -> None:
        if config.imputation is None and values.isnan().any():
            channel = channels[int(values.isnan().nonzero()[0, -1])]
            raise ValueError(
                f"source {source!r}, client {client!r}, channel {channel!r}: "
                "missing values require data.scaler.imputation='masked'"
            )
