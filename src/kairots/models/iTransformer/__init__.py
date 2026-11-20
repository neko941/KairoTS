"""iTransformer (Liu et al., ICLR 2024), for multivariate forecasting.

Reference: thuml/iTransformer@c2426e68ca13f74aaec08045c5c724d8ad328124.
It uses inverted variates, noncausal attention, RevIN, and shared horizon projection.
Blocks use the paper's two LayerNorms; no final encoder normalization is added.
"""

from dataclasses import dataclass

from kairots.models.iTransformer.InvertedTransformer import InvertedTransformer


class iTransformer(InvertedTransformer):  # noqa: N801
    """Paper-faithful inverted Transformer forecasting architecture."""

    TASKS = ("Forecasting",)
    COVARIATES = ("past", "calendar")

    @dataclass(kw_only=True)
    class Config(InvertedTransformer.Config):
        """iTransformer forecast settings."""

        name: str = "iTransformer"
