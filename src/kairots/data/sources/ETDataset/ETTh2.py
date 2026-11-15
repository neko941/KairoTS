"""Hourly ETT variant 2."""

from dataclasses import dataclass

from kairots.data.sources.ETDataset import ETDataset


class ETTh2(ETDataset):
    FILE_NAME = "ETTh2"
    FREQUENCY = "hourly"
    SHA256 = "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"

    @dataclass(kw_only=True)
    class Config(ETDataset.Config):
        name: str = "ETTh2"
