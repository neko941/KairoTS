"""Structure kind for samples observed at their own, uneven times."""

from dataclasses import dataclass

from kairots.data.sources.Panel import Panel


@dataclass(kw_only=True)
class Irregular(Panel):
    """A panel whose rows are a sample's observation times, not a shared grid.

    The first channel is `time`, the observation time of the row. The other channels
    are NaN where that variable was not observed, so the mask is `~isnan`. A sample
    with fewer than `input_length` observation times is padded with rows that are NaN
    in every channel, `time` included. Models must read the mask (GRUD does); one
    that does not sees NaN.
    """

    ABSTRACT = True
    REQUIRES = {"data.scaler.imputation": ("masked",)}
