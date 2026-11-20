"""Quarter-hourly ETT variant 2."""

from dataclasses import dataclass

from kairots.data.sources.ETDataset import ETDataset


class ETTm2(ETDataset):
    FILE_NAME = "ETTm2"
    FREQUENCY = "15min"
    SHA256 = "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd"

    @dataclass(kw_only=True)
    class Config(ETDataset.Config):
        name: str = "ETTm2"
