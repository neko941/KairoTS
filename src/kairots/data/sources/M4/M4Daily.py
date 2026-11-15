"""M4 Daily: 4,227 series with a 14-step test horizon."""

from dataclasses import dataclass

from kairots.data.sources.M4 import M4


@dataclass(kw_only=True)
class M4Daily(M4):
    TASKS = ("Forecasting",)
    PERIOD = "Daily"
    SERIES_COUNT = 4227
    HORIZON = 14
    SEASONALITY = 1
    SHA256 = {
        "train": "78e94591c60c06309f1e544fd7b2ccba28f3616b01913dd336a1d1d98483a1ec",
        "test": "633b0815626ae266d36872c4d53e95eb93ee8d43e79587fceec519eba3e6b05e",
    }

    @dataclass(kw_only=True)
    class Config(M4.Config):
        name: str = "M4Daily"
        seasonality: int = 1
