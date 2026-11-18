"""iTransformer (Liu et al., ICLR 2024), for multivariate forecasting.

Reference: `thuml/iTransformer` at commit
`c2426e68ca13f74aaec08045c5c724d8ad328124`. The architecture uses inverted
variates, full noncausal attention, per-window RevIN, and a shared horizon
projection. Its encoder follows the paper's block-level layer norms without an
additional encoder-final normalization.
"""

from dataclasses import dataclass

from kairots.models.iTransformer.InvertedTransformer import InvertedTransformer


class iTransformer(InvertedTransformer):  # noqa: N801
    """Paper-faithful inverted Transformer forecasting architecture."""

    @dataclass(kw_only=True)
    class Config(InvertedTransformer.Config):
        """iTransformer forecast settings."""

        name: str = "iTransformer"
