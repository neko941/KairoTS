"""Hourly ETT variant 1."""

from dataclasses import dataclass

from kairots.data.sources.ETDataset import ETDataset


class ETTh1(ETDataset):
    FILE_NAME = "ETTh1"
    FREQUENCY = "hourly"
    SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"

    @dataclass(kw_only=True)
    class Config(ETDataset.Config):
        name: str = "ETTh1"
