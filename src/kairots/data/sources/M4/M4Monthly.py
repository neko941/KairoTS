"""M4 Monthly: 48,000 series with an 18-step test horizon."""

from dataclasses import dataclass

from kairots.data.sources.M4 import M4


@dataclass(kw_only=True)
class M4Monthly(M4):
    TASKS = ("Forecasting",)
    PERIOD = "Monthly"
    SERIES_COUNT = 48000
    HORIZON = 18
    SEASONALITY = 12
    SHA256 = {
        "train": "63aaa56198b4a22279a2fec541f56921f3bed6d8e0f84e550206df9276f2e9b9",
        "test": "5e3538bf23b4a09f4078180c3e2b34859f0382e1c4f22564affc1797e8f3f48f",
    }

    @dataclass(kw_only=True)
    class Config(M4.Config):
        name: str = "M4Monthly"
        seasonality: int = 12
