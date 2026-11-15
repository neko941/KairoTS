"""Collection of equal-length labeled multivariate sequences (classification data)."""

from dataclasses import dataclass

from kairots.data.sources.base import Source


@dataclass(kw_only=True)
class Panel(Source):
    """Each sample is its own sequence of `input_length` steps with one class label.

    Client frames have one row per (sample, step), sorted by both, with columns
    `sample`, `step`, `fold` (the archive's official "train"/"test"), `label` (index
    into the shared `classes` vocabulary in the manifest) and the channel columns.
    """

    ABSTRACT = True

    @dataclass(kw_only=True)
    class Config(Source.Config):
        input_length: int = 0
        num_classes: int = 0
