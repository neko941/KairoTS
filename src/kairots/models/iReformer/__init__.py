"""iReformer: the LSH Reformer iTransformer variant in Liu et al. (2024)."""

from dataclasses import dataclass

from torch import nn

from kairots.models.iReformer.Reformer import Reformer
from kairots.models.iTransformer.InvertedTransformer import InvertedTransformer


class iReformer(InvertedTransformer):  # noqa: N801
    """Invert Reformer LSH attention over variate tokens for forecasting."""

    TASKS = ("Forecasting",)
    COVARIATES = ("past", "calendar")

    @dataclass(kw_only=True)
    class Config(InvertedTransformer.Config):
        """Shared iTransformer forecast settings with the iReformer name."""

        name: str = "iReformer"

    def _attention(self, *, config: InvertedTransformer.Config) -> nn.Module:
        return Reformer(d_model=config.d_model, heads=config.attention_heads)

    def _encoder_norm(self, *, config: InvertedTransformer.Config) -> nn.Module:
        return nn.LayerNorm(config.d_model)
