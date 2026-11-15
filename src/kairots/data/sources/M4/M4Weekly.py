"""M4 Weekly: 359 series with a 13-step test horizon."""

from dataclasses import dataclass

from kairots.data.sources.M4 import M4


@dataclass(kw_only=True)
class M4Weekly(M4):
    TASKS = ("Forecasting",)
    PERIOD = "Weekly"
    SERIES_COUNT = 359
    HORIZON = 13
    SEASONALITY = 1
    SHA256 = {
        "train": "d478d3f6ed673e6ed3c2cb0fbed5425f72eedd51318ebebeb0b5ecfcefc37714",
        "test": "3b2bc4be8e636e802260dcd843ca3c5da40536a1fbe1ecb3229a97dae4f44688",
    }

    @dataclass(kw_only=True)
    class Config(M4.Config):
        name: str = "M4Weekly"
        seasonality: int = 1
