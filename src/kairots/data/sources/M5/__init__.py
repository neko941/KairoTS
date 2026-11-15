"""M5 Accuracy competition data and its product-store bottom-level series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.M5._common import (
    DIRECTORY_URL,
    FILE_IDS,
    IDENTIFIERS,
    SERIES_COUNT,
    SHA256,
    read_sales,
)
from kairots.data.sources.Multivariate import Multivariate

CHANNELS = tuple(f"series_{index:05d}" for index in range(SERIES_COUNT))
RAW_FILES = (
    "calendar.csv",
    "sales_train_validation.csv",
    "sales_test_validation.csv",
    "weights_validation.csv",
    "sales_train_evaluation.csv",
    "sales_test_evaluation.csv",
    "weights_evaluation.csv",
    "sell_prices.csv",
)


@dataclass(kw_only=True)
class M5(Multivariate):
    ABSTRACT = True
    FOLD: ClassVar[str] = ""
    TRAIN_DAYS: ClassVar[int] = 0
    SERIES_COUNT: ClassVar[int] = SERIES_COUNT
    FORECAST_LENGTH: ClassVar[int] = 28

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "M5"
        raw_path: str = "datasets/M5/raw"
        input_channels: tuple[str, ...] = CHANNELS
        batch_size: int = 1
        seasonality: int = 7

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        return [Path(config.raw_path) / name for name in RAW_FILES]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        return tuple(
            RemoteFile(
                path=name,
                urls=(DIRECTORY_URL.format(FILE_IDS[name]),),
                sha256=SHA256[name],
            )
            for name in RAW_FILES
        )

    @classmethod
    def load(cls, *, config: Config) -> M5:
        if (
            len(config.input_channels) != cls.SERIES_COUNT
            or len(set(config.input_channels)) != cls.SERIES_COUNT
        ):
            raise ValueError(
                "M5 requires the complete ordered set of 30,490 bottom series"
            )
        root = Path(config.raw_path)
        train = read_sales(
            path=root / f"sales_train_{cls.FOLD}.csv",
            expected_days=cls.TRAIN_DAYS,
            series_count=cls.SERIES_COUNT,
        )
        test = read_sales(
            path=root / f"sales_test_{cls.FOLD}.csv",
            expected_days=cls.FORECAST_LENGTH,
            series_count=cls.SERIES_COUNT,
        )
        if train.identifiers != test.identifiers:
            raise ValueError(
                "M5 train and test rows have different series IDs or order"
            )
        first_test_day = int(train.days[-1].removeprefix("d_")) + 1
        if [int(day.removeprefix("d_")) for day in test.days] != list(
            range(first_test_day, first_test_day + cls.FORECAST_LENGTH)
        ):
            raise ValueError("M5 train and test folds are not contiguous")
        values = np.concatenate((train.values.T, test.values.T), axis=0)
        frame = pl.DataFrame(values, schema=config.input_channels, orient="row")
        frame = frame.with_columns(
            pl.Series("date", np.arange(len(values))),
            pl.Series(
                "fold",
                ["train"] * cls.TRAIN_DAYS + ["test"] * cls.FORECAST_LENGTH,
            ),
        )
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={},
            clients=[("M5", frame)],
            manifest_fields={
                "frequency": "daily",
                "forecast_horizon": cls.FORECAST_LENGTH,
                "series": cls.SERIES_COUNT,
                "hierarchy": "M5 12-level unit-sales; WRMSSE bottom-up aggregation",
                "raw_files": [str(path) for path in cls.raw_files(config=config)],
                "series_metadata": {
                    channel: dict(zip(IDENTIFIERS, identifiers, strict=True))
                    for channel, identifiers in zip(
                        config.input_channels, train.identifiers, strict=True
                    )
                },
            },
        )
