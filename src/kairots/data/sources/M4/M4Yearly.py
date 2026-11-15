"""M4 Yearly: 23,000 series with a six-step test horizon."""

from dataclasses import dataclass

from kairots.data.sources.M4 import M4


@dataclass(kw_only=True)
class M4Yearly(M4):
    TASKS = ("Forecasting",)
    PERIOD = "Yearly"
    SERIES_COUNT = 23000
    HORIZON = 6
    SEASONALITY = 1
    SHA256 = {
        "train": "6aa668d4f04654c0bf5f07b266a51c556f0df2faf4709f852653755396a249e4",
        "test": "6dc2dd45b3af320389e2d3735cb4b3cdf320b63cbcda15d1a17193ab99702306",
    }

    @dataclass(kw_only=True)
    class Config(M4.Config):
        name: str = "M4Yearly"
        seasonality: int = 1
