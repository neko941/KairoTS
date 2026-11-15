"""Forecast an official test fold from the history before it.

TSLib M4 `last_insample_window`: test targets tile the test fold; validation
targets tile the last `validation_windows * output_length` steps of train; train
windows fill the rest. Inputs may reach back across a fold boundary. Unlike
TSLib, which validates on the test set, validation is held out from train. It is
off by default (`validation_windows=0`).
"""

from dataclasses import dataclass

import polars as pl

from kairots.data.splits.base import Strided, Window


@dataclass(kw_only=True)
class TestFold:
    @dataclass(kw_only=True)
    class Config(Window.Config, Strided.Config):
        name: str = "TestFold"
        output_length: int = 48
        validation_windows: int = 0
        min_windows: int = 1

    def __init__(self, *, config: Config | None = None) -> None:
        self.config = config or self.Config()

    def series(
        self,
        *,
        frame: pl.DataFrame,
        input_length: int,
        output_length: int,
        offset_length: int,
    ) -> dict[str, list[tuple[int, int]]]:
        folds = frame.get_column("fold").to_list()
        train_rows = folds.count("train")
        if folds[:train_rows] != ["train"] * train_rows:
            raise ValueError(
                "TestFold series needs the train fold before the test fold"
            )
        cut = train_rows - self.config.validation_windows * output_length
        size = input_length + offset_length + output_length
        offset = input_length + offset_length
        train = [
            (start, start + offset)
            for start in range(0, cut - size + 1, self.config.train_stride)
        ]
        validation = [
            (target - offset, target)
            for target in range(cut, train_rows, output_length)
            if target + output_length <= train_rows and target - offset >= 0
        ]
        test = [
            (target - offset, target)
            for target in range(train_rows, len(folds), output_length)
            if target + output_length <= len(folds) and target - offset >= 0
        ]
        return {"train": train, "validation": validation, "test": test}
