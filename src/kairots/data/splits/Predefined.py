"""Keep an archive's official train/test folds; carve validation from train.

Validation is off by default (`validation_ratio=0`); anomaly configs set it,
because their threshold is calibrated on validation scores.

Panel data (`split`): validation is drawn per class (stratified) with a seeded
shuffle, so every class present in a client's train fold keeps the same share in
validation.

Series data (`series`): the last `validation_ratio` of the train fold is
validation. Train windows step by `train_stride` (0: the window size); validation
and test windows tile their fold without overlap, so every evaluated step is scored
exactly once. Unlike TSLib, whose validation is the last 20% of train and is also
trained on, validation here is held out.
"""

import random
from dataclasses import dataclass

import polars as pl

from kairots.data.splits.base import Strided


@dataclass(kw_only=True)
class Predefined:
    @dataclass(kw_only=True)
    class Config(Strided.Config):
        name: str = "Predefined"
        validation_ratio: float = 0.0
        seed: int = 0
        min_windows: int = 1
        window_length: int = 0
        # Series only: window length (TSLib anomaly seq_len) and the step between
        # train windows; 0 tiles them like evaluation
        # windows (TSLib anomaly loaders use step = window size).
        train_stride: int = 0

    def __init__(self, *, config: Config | None = None) -> None:
        self.config = config or self.Config()

    def split(self, *, samples: pl.DataFrame) -> dict[str, list[int]]:
        """Map each split to local sample indexes (row positions in `samples`).

        Args:
            samples: One row per sample, in local order, with `fold` and `label`.
        """
        folds = samples.get_column("fold").to_list()
        labels = samples.get_column("label").to_list()
        generator = random.Random(self.config.seed)
        validation: list[int] = []
        for label in sorted(set(labels)):
            members = [
                index
                for index, (fold, value) in enumerate(zip(folds, labels, strict=True))
                if fold == "train" and value == label
            ]
            generator.shuffle(members)
            # Floor, so a class with one train sample keeps it in train.
            validation += members[: int(len(members) * self.config.validation_ratio)]
        held = set(validation)
        return {
            "train": [
                index
                for index, fold in enumerate(folds)
                if fold == "train" and index not in held
            ],
            "validation": sorted(validation),
            "test": [index for index, fold in enumerate(folds) if fold == "test"],
        }

    def series(
        self,
        *,
        frame: pl.DataFrame,
        input_length: int,
        output_length: int,
        offset_length: int,
    ) -> dict[str, list[tuple[int, int]]]:
        """(start, target) windows that never cross a fold or the validation cut."""
        folds = frame.get_column("fold").to_list()
        train_rows = folds.count("train")
        if folds[:train_rows] != ["train"] * train_rows:
            raise ValueError(
                "Predefined series needs the train fold before the test fold"
            )
        size = input_length + offset_length + output_length
        cut = train_rows - int(train_rows * self.config.validation_ratio)

        def windows(*, first: int, end: int, stride: int) -> list[tuple[int, int]]:
            return [
                (start, start + input_length + offset_length)
                for start in range(first, end - size + 1, stride)
            ]

        return {
            "train": windows(first=0, end=cut, stride=self.config.train_stride or size),
            "validation": windows(first=cut, end=train_rows, stride=size),
            "test": windows(first=train_rows, end=len(folds), stride=size),
        }
