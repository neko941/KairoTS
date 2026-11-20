"""GRU-D's k-fold cross-validation over the train fold (Che et al., 2018).

The official code cuts the samples, in order and without shuffling, into `folds`
contiguous chunks (scikit-learn `KFold(shuffle=False)`). Fold `test_fold` tests on chunk
`test_fold`, validates on the next chunk (wrapping) and trains on the rest. The test
fold is unused, and each client is cut separately.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from kairots.data.splits.Predefined import Predefined


class KFold(Predefined):
    @dataclass(kw_only=True)
    class Config(Predefined.Config):
        name: str = "KFold"
        folds: int = 5
        test_fold: int = 0

    def __init__(self, *, config: Config | None = None) -> None:
        self.config = config or self.Config()

    def split(self, *, samples: pl.DataFrame) -> dict[str, list[int]]:
        if not 0 <= self.config.test_fold < self.config.folds:
            raise ValueError("KFold.test_fold must be in [0, folds)")
        pool = np.flatnonzero(samples.get_column("fold").to_numpy() == "train")
        chunks = np.array_split(pool, self.config.folds)
        test = self.config.test_fold
        validation = (test + 1) % self.config.folds
        return {
            "train": np.concatenate(
                [
                    chunk
                    for index, chunk in enumerate(chunks)
                    if index not in (test, validation)
                ]
            ).tolist(),
            "validation": chunks[validation].tolist(),
            "test": chunks[test].tolist(),
        }
