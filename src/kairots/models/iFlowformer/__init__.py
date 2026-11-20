"""iFlowformer: the FlowAttention iTransformer variant in Liu et al. (2024)."""

from dataclasses import dataclass

from torch import nn

from kairots.layers.FlowAttention import FlowAttention
from kairots.models.iTransformer.InvertedTransformer import InvertedTransformer


class iFlowformer(InvertedTransformer):  # noqa: N801
    """Invert Flowformer attention over variate tokens for forecasting."""

    TASKS = ("Forecasting",)
    COVARIATES = ("past", "calendar")

    @dataclass(kw_only=True)
    class Config(InvertedTransformer.Config):
        """Shared iTransformer forecast settings with the iFlowformer name."""

        name: str = "iFlowformer"

    def _attention(self, *, config: InvertedTransformer.Config) -> nn.Module:
        return FlowAttention(d_model=config.d_model, heads=config.attention_heads)

    def _encoder_norm(self, *, config: InvertedTransformer.Config) -> nn.Module:
        return nn.LayerNorm(config.d_model)
