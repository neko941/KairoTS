"""M5 final-evaluation fold: 1,941 train and 28 test days."""

from dataclasses import dataclass

from kairots.data.sources.M5 import M5


@dataclass(kw_only=True)
class M5Evaluation(M5):
    TASKS = ("Forecasting",)
    FOLD = "evaluation"
    TRAIN_DAYS = 1_941

    @dataclass(kw_only=True)
    class Config(M5.Config):
        name: str = "M5Evaluation"
