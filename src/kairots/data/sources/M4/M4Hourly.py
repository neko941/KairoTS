"""M4 Hourly: 414 series, 700 to 960 training steps, 48-step test, season 24."""

from dataclasses import dataclass

from kairots.data.sources.M4 import M4


@dataclass(kw_only=True)
class M4Hourly(M4):
    TASKS = ("Forecasting",)
    PERIOD = "Hourly"
    SERIES_COUNT = 414
    HORIZON = 48
    SEASONALITY = 24
    SHA256 = {
        "train": "ea59b7783573c49077a835ab6465c7d66f1474783360f310988a9a737fbca62f",
        "test": "71a57fccb15e534d973626ada4fb87febf789e9317715b7cc6dd3b5f90db6f42",
    }

    @dataclass(kw_only=True)
    class Config(M4.Config):
        name: str = "M4Hourly"
        seasonality: int = 24
