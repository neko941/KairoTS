"""M4 competition (Makridakis et al., 2020); CSVs from Mcompetitions/M4-methods."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Collection import Collection

SOURCE_REVISION = "6c1067e5a57161249b17289a565178dc7a3fb3ca"
URL_ROOT = (
    "https://raw.githubusercontent.com/Mcompetitions/M4-methods/"
    f"{SOURCE_REVISION}/Dataset"
)
INFO_SHA256 = "26a75cc9b9f65e351b39b0bdbb7e3464ac837df0baf541ab55abdd2ba1a14c88"


@dataclass(kw_only=True)
class M4(Collection):
    ABSTRACT = True
    PERIOD: ClassVar[str] = ""
    SHA256: ClassVar[dict[str, str]] = {}
    SERIES_COUNT: ClassVar[int] = 0
    HORIZON: ClassVar[int] = 0
    SEASONALITY: ClassVar[int] = 1

    @dataclass(kw_only=True)
    class Config(Collection.Config):
        name: str = "M4"
        raw_path: str = "datasets/M4/raw"

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        return [
            Path(config.raw_path) / fold / f"{cls.PERIOD}-{fold.lower()}.csv"
            for fold in ("Train", "Test")
        ] + [Path(config.raw_path) / "M4-info.csv"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        return tuple(
            RemoteFile(
                path=f"{fold}/{cls.PERIOD}-{fold.lower()}.csv",
                urls=(f"{URL_ROOT}/{fold}/{cls.PERIOD}-{fold.lower()}.csv",),
                sha256=cls.SHA256[fold.lower()],
            )
            for fold in ("Train", "Test")
        ) + (
            RemoteFile(
                path="M4-info.csv",
                urls=(f"{URL_ROOT}/M4-info.csv",),
                sha256=INFO_SHA256,
            ),
        )

    @classmethod
    def load(cls, *, config: Config) -> M4:
        train_path, test_path, info_path = cls.raw_files(config=config)
        with info_path.open(newline="", encoding="utf-8") as info_file:
            metadata_rows = [
                row for row in csv.DictReader(info_file) if row["SP"] == cls.PERIOD
            ]
        metadata = {row["M4id"]: row for row in metadata_rows}
        if len(metadata) != len(metadata_rows):
            raise ValueError(f"M4 {cls.PERIOD} metadata has duplicate series IDs")
        if len(metadata_rows) != cls.SERIES_COUNT:
            raise ValueError(
                f"M4 {cls.PERIOD} metadata has {len(metadata_rows)} series; "
                f"expected {cls.SERIES_COUNT}"
            )
        if any(
            int(row["Horizon"]) != cls.HORIZON
            or int(row["Frequency"]) != cls.SEASONALITY
            for row in metadata.values()
        ):
            raise ValueError(f"M4 {cls.PERIOD} metadata has wrong horizon/frequency")

        def values(*, row: list[str], series_id: str, fold: str) -> np.ndarray:
            data = row[1:]
            while data and not data[-1].strip():
                data.pop()
            if not data or any(not item.strip() for item in data):
                raise ValueError(f"M4 {series_id} has empty {fold} observations")
            result = np.asarray([float(item) for item in data], dtype=np.float32)
            if not np.isfinite(result).all():
                raise ValueError(f"M4 {series_id} has non-finite {fold} values")
            return result

        clients: list[tuple[str, pl.DataFrame]] = []
        seen: set[str] = set()
        with train_path.open(newline="", encoding="utf-8") as train_file:
            train_rows = csv.reader(train_file)
            next(train_rows, None)
            with test_path.open(newline="", encoding="utf-8") as test_file:
                test_rows = csv.reader(test_file)
                next(test_rows, None)
                for train_row, test_row in zip(train_rows, test_rows, strict=True):
                    series_id = train_row[0]
                    if (
                        not series_id
                        or series_id in seen
                        or test_row[0] != series_id
                        or series_id not in metadata
                    ):
                        raise ValueError(
                            f"M4 series ID mismatch, duplicate or missing metadata: "
                            f"{series_id!r} != {test_row[0]!r}"
                        )
                    seen.add(series_id)
                    train_values = values(
                        row=train_row, series_id=series_id, fold="train"
                    )
                    test_values = values(row=test_row, series_id=series_id, fold="test")
                    if len(test_values) != cls.HORIZON:
                        raise ValueError(
                            f"M4 {series_id} has {len(test_values)} test values; "
                            f"expected {cls.HORIZON}"
                        )
                    series_values = np.concatenate((train_values, test_values))
                    clients.append(
                        (
                            series_id,
                            pl.DataFrame(
                                {
                                    "date": np.arange(len(series_values)),
                                    "fold": ["train"] * len(train_values)
                                    + ["test"] * len(test_values),
                                    "value": series_values,
                                }
                            ),
                        )
                    )
        if seen != set(metadata):
            raise ValueError(
                f"M4 {cls.PERIOD} train rows omit metadata IDs: "
                f"{len(set(metadata) - seen)} omitted"
            )
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={},
            clients=clients,
            manifest_fields={
                "frequency": cls.PERIOD,
                "seasonality": cls.SEASONALITY,
                "forecast_horizon": cls.HORIZON,
                "raw_files": [str(train_path), str(test_path), str(info_path)],
                "series": len(clients),
                "series_metadata": {
                    series_id: {
                        "category": row["category"],
                        "frequency": int(row["Frequency"]),
                        "starting_date": row["StartingDate"],
                    }
                    for series_id, row in metadata.items()
                },
            },
        )
