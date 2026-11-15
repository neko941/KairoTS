"""M5 public-leaderboard validation fold: 1,913 train and 28 test days."""

from dataclasses import dataclass

from kairots.data.sources.M5 import M5


@dataclass(kw_only=True)
class M5Validation(M5):
    TASKS = ("Forecasting",)
    FOLD = "validation"
    TRAIN_DAYS = 1_913

    @dataclass(kw_only=True)
    class Config(M5.Config):
        name: str = "M5Validation"
