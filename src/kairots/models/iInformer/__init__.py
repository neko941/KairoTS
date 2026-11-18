"""iInformer: the ProbSparse iTransformer variant in Liu et al. (2024)."""

from dataclasses import dataclass

from torch import nn

from kairots.layers.Attention import Attention
from kairots.models.iTransformer.InvertedTransformer import InvertedTransformer


class iInformer(InvertedTransformer):  # noqa: N801
    """Invert Informer ProbSparse attention over variate tokens for forecasting."""

    @dataclass(kw_only=True)
    class Config(InvertedTransformer.Config):
        """Shared iTransformer forecast settings with the author factor default."""

        name: str = "iInformer"
        sampling_factor: int = 1

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.sampling_factor < 1:
                raise ValueError("iInformer sampling_factor must be positive")

    def _attention(self, *, config: InvertedTransformer.Config) -> nn.Module:
        return Attention(
            d_model=config.d_model,
            heads=config.attention_heads,
            sparse=True,
            sampling_factor=getattr(config, "sampling_factor", 1),
            dropout=config.dropout,
        )

    def _encoder_norm(self, *, config: InvertedTransformer.Config) -> nn.Module:
        return nn.LayerNorm(config.d_model)
