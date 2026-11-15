"""Regular-grid multivariate source family."""

from dataclasses import dataclass

from kairots.data.sources.base import Source


@dataclass(kw_only=True)
class Multivariate(Source):
    """Source with aligned channels on one regular time grid."""

    ABSTRACT = True
