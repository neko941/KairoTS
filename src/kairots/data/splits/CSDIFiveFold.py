"""CSDI PhysioNet patient-level five-fold split with held-out validation."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from kairots.data.splits.Predefined import Predefined


class CSDIFiveFold(Predefined):
    @dataclass(kw_only=True)
    class Config(Predefined.Config):
        name: str = "CSDIFiveFold"
        input_length: int = 0
        output_length: int = 48
        fold: int = 0
        validation_ratio: float = 0.125
        seed: int = 1

    def __init__(self, *, config: Config | None = None) -> None:
        self.config = config or self.Config()

    def split(self, *, samples: pl.DataFrame) -> dict[str, list[int]]:
        if not 0 <= self.config.fold < 5:
            raise ValueError("CSDIFiveFold.fold must be between 0 and 4")
        count = samples.height
        if count % 5:
            raise ValueError(
                f"CSDIFiveFold requires a multiple of 5 patients, got {count}"
            )
        permutation = np.random.RandomState(self.config.seed).permutation(count)
        fold_size = count // 5
        first = self.config.fold * fold_size
        test = permutation[first : first + fold_size].tolist()
        train_validation = np.concatenate(
            (permutation[:first], permutation[first + fold_size :])
        )
        train_validation = np.random.RandomState(self.config.seed).permutation(
            train_validation
        )
        train_size = len(train_validation) - int(
            len(train_validation) * self.config.validation_ratio
        )
        return {
            "train": train_validation[:train_size].tolist(),
            "validation": train_validation[train_size:].tolist(),
            "test": test,
        }
