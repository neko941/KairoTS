"""M4 Quarterly: 24,000 series with an eight-step test horizon."""

from dataclasses import dataclass

from kairots.data.sources.M4 import M4


@dataclass(kw_only=True)
class M4Quarterly(M4):
    TASKS = ("Forecasting",)
    PERIOD = "Quarterly"
    SERIES_COUNT = 24000
    HORIZON = 8
    SEASONALITY = 4
    SHA256 = {
        "train": "4450678cd493aa965c14bde984a5b62a66d8c3b2f7af1e53779bc91ff008ccdc",
        "test": "bf1cab3b969a744c220cdc148e3829803eaa5da2754270fc4f376e775cd7d0ae",
    }

    @dataclass(kw_only=True)
    class Config(M4.Config):
        name: str = "M4Quarterly"
        seasonality: int = 4
