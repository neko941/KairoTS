"""M3 Other: 174 series with the paper's eight-step held-out horizon."""

from dataclasses import dataclass

from kairots.data.sources.M3 import M3


@dataclass(kw_only=True)
class M3Other(M3):
    TASKS = ("Forecasting",)
    FREQUENCY = "Other"
    SHEET = "M3Other"
    SERIES_COUNT = 174
    HORIZON = 8

    @dataclass(kw_only=True)
    class Config(M3.Config):
        name: str = "M3Other"
        seasonality: int = 1
