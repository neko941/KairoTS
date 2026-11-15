"""M3 Quarterly: 756 series with the paper's eight-step held-out horizon."""

from dataclasses import dataclass

from kairots.data.sources.M3 import M3


@dataclass(kw_only=True)
class M3Quarterly(M3):
    TASKS = ("Forecasting",)
    FREQUENCY = "Quarterly"
    SHEET = "M3Quart"
    SERIES_COUNT = 756
    HORIZON = 8
    START_FIELDS = ("start_year", "start_quarter")

    @dataclass(kw_only=True)
    class Config(M3.Config):
        name: str = "M3Quarterly"
        seasonality: int = 4
