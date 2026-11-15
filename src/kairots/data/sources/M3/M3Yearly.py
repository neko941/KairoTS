"""M3 Yearly: 645 series with the paper's six-step held-out horizon."""

from dataclasses import dataclass

from kairots.data.sources.M3 import M3


@dataclass(kw_only=True)
class M3Yearly(M3):
    TASKS = ("Forecasting",)
    FREQUENCY = "Yearly"
    SHEET = "M3Year"
    SERIES_COUNT = 645
    HORIZON = 6
    START_FIELDS = ("start_year",)

    @dataclass(kw_only=True)
    class Config(M3.Config):
        name: str = "M3Yearly"
        seasonality: int = 1
