"""iFlashformer: the FlashAttention iTransformer variant in Liu et al. (2024)."""

from dataclasses import dataclass

from torch import nn

from kairots.layers.FlashAttention import FlashAttention
from kairots.models.iTransformer.InvertedTransformer import InvertedTransformer


class iFlashformer(InvertedTransformer):  # noqa: N801
    """Invert Flashformer attention over variate tokens for forecasting."""

    @dataclass(kw_only=True)
    class Config(InvertedTransformer.Config):
        """Shared iTransformer forecast settings with the iFlashformer name."""

        name: str = "iFlashformer"

    def _attention(self, *, config: InvertedTransformer.Config) -> nn.Module:
        return FlashAttention(d_model=config.d_model, heads=config.attention_heads)

    def _encoder_norm(self, *, config: InvertedTransformer.Config) -> nn.Module:
        return nn.LayerNorm(config.d_model)
