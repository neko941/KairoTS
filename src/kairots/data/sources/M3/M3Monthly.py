"""M3 Monthly: 1,428 series with the paper's 18-step held-out horizon."""

from dataclasses import dataclass

from kairots.data.sources.M3 import M3


@dataclass(kw_only=True)
class M3Monthly(M3):
    TASKS = ("Forecasting",)
    FREQUENCY = "Monthly"
    SHEET = "M3Month"
    SERIES_COUNT = 1428
    HORIZON = 18
    START_FIELDS = ("start_year", "start_month")

    @dataclass(kw_only=True)
    class Config(M3.Config):
        name: str = "M3Monthly"
        seasonality: int = 12
