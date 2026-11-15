"""Historical TSLib validation windows that overlap the training fold."""

from dataclasses import dataclass

import polars as pl

from kairots.data.splits.Predefined import Predefined


@dataclass(kw_only=True)
class HistoricalValidation(Predefined):
    @dataclass(kw_only=True)
    class Config(Predefined.Config):
        name: str = "HistoricalValidation"

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)

    def series(
        self,
        *,
        frame: pl.DataFrame,
        input_length: int,
        output_length: int,
        offset_length: int,
    ) -> dict[str, list[tuple[int, int]]]:
        splits = super().series(
            frame=frame,
            input_length=input_length,
            output_length=output_length,
            offset_length=offset_length,
        )
        folds = frame.get_column("fold").to_list()
        train_rows = folds.count("train")
        size = input_length + offset_length + output_length
        stride = self.config.train_stride or size
        splits["train"] = [
            (start, start + input_length + offset_length)
            for start in range(0, train_rows - size + 1, stride)
        ]
        return splits
