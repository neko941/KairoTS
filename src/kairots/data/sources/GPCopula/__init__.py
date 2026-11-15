"""Base contract for the pinned GP-Copula paper datasets."""

from dataclasses import dataclass

from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class GPCopula(Multivariate):
    ABSTRACT = True

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "GPCopula"
        raw_path: str = "datasets/GPCopula/raw"
