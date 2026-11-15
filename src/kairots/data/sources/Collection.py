"""Collection kind for differently sized univariate series."""

from dataclasses import dataclass

from kairots.data.sources.base import Source


@dataclass(kw_only=True)
class Collection(Source):
    """Collection of univariate series of different lengths, one client per series.
    Client frames have an integer `date` (step index), a `fold` (the archive's
    official `train`/`test`) and one channel column `value`.
    """

    ABSTRACT = True

    @dataclass(kw_only=True)
    class Config(Source.Config):
        input_channels: tuple[str, ...] = ("value",)
