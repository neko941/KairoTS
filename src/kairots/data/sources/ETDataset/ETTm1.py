"""Quarter-hourly ETT variant 1."""

from dataclasses import dataclass

from kairots.data.sources.ETDataset import ETDataset


class ETTm1(ETDataset):
    FILE_NAME = "ETTm1"
    FREQUENCY = "15min"
    SHA256 = "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"

    @dataclass(kw_only=True)
    class Config(ETDataset.Config):
        name: str = "ETTm1"
