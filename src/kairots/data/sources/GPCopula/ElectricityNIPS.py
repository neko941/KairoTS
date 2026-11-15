"""Hourly GP-Copula Electricity release used by TimeGrad's paper experiment."""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.GPCopula import GPCopula

REVISION = "442bd4ffffa4a0fcf9ae7aa25db9632fbe58a7ea"
URL = (
    "https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/"
    f"{REVISION}/datasets/electricity_nips.tar.gz"
)
SHA256 = "b80a63a4963f487dc3b2a3be8cb17b3c1e54f71fddae9b5a12967b38c990c4cf"
SERIES = 370
TRAIN_LENGTH = 5833
HORIZON = 24
ROLLING_DATES = 7


@dataclass(kw_only=True)
class ElectricityNIPS(GPCopula):
    TASKS = ("SampleForecasting",)

    @dataclass(kw_only=True)
    class Config(GPCopula.Config):
        name: str = "ElectricityNIPS"
        input_channels: tuple[str, ...] = tuple(str(index) for index in range(SERIES))
        seasonality: int = 24

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        return [Path(config.raw_path) / "electricity_nips.tar.gz"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        return (RemoteFile(path="electricity_nips.tar.gz", urls=(URL,), sha256=SHA256),)

    @classmethod
    def load(cls, *, config: Config) -> ElectricityNIPS:
        (path,) = cls.raw_files(config=config)
        if _sha256(path=path) != SHA256:
            raise ValueError(
                f"{path} does not match the pinned ElectricityNIPS archive"
            )
        if config.input_channels != tuple(str(index) for index in range(SERIES)):
            raise ValueError("ElectricityNIPS requires the pinned 0..369 channel order")
        with tarfile.open(path, mode="r:gz") as archive:
            train_entries = _entries(
                archive=archive, member="electricity_nips/train/data.json"
            )
            test_entries = _entries(
                archive=archive, member="electricity_nips/test/data.json"
            )
        frame, timestamps, train_length = _build_frame(
            train_entries=train_entries,
            test_entries=test_entries,
            channels=config.input_channels,
            output_length=HORIZON,
            rolling_dates=ROLLING_DATES,
        )
        if train_length != TRAIN_LENGTH:
            raise ValueError(
                f"ElectricityNIPS has {train_length} training steps; "
                f"expected {TRAIN_LENGTH}"
            )
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={path.stem: timestamps},
            clients=[(path.stem, frame)],
            manifest_fields={
                "raw_files": [str(path)],
                "frequency": "hourly",
                "dataset_revision": REVISION,
                "series": SERIES,
                "train_length": train_length,
                "prediction_length": HORIZON,
                "rolling_test_dates": ROLLING_DATES,
                "train_padding": "per-series mean (GluonTS MultivariateGrouper)",
                "test_padding": "zero (GluonTS MultivariateGrouper)",
            },
        )


def _entries(*, archive: tarfile.TarFile, member: str) -> list[dict[str, Any]]:
    file = archive.extractfile(member)
    if file is None:
        raise ValueError(f"GP-Copula archive is missing {member}")
    return [json.loads(line) for line in file]


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for block in iter(lambda: archive.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_frame(
    *,
    train_entries: list[dict[str, Any]],
    test_entries: list[dict[str, Any]],
    channels: tuple[str, ...],
    output_length: int,
    rolling_dates: int,
) -> tuple[pl.DataFrame, tuple[datetime, ...], int]:
    """Group the released univariate JSON records as the pinned GluonTS grouper does."""
    channel_count = len(channels)
    if channel_count == 0 or output_length < 1 or rolling_dates < 1:
        raise ValueError(
            "GP-Copula grouping needs channels and positive output lengths"
        )
    train_by_channel: dict[int, dict[str, Any]] = {}
    for entry in train_entries:
        channel = _channel(entry=entry, channel_count=channel_count)
        if channel in train_by_channel:
            raise ValueError(f"duplicate train series {channel}")
        train_by_channel[channel] = entry
    if set(train_by_channel) != set(range(channel_count)):
        raise ValueError("GP-Copula train series do not cover every channel")

    train_starts = {
        channel: _start(entry=entry) for channel, entry in train_by_channel.items()
    }
    first = min(train_starts.values())
    train_offsets = {
        channel: _steps(start=start, first=first)
        for channel, start in train_starts.items()
    }
    train_last = max(
        train_offsets[channel] + len(entry["target"]) - 1
        for channel, entry in train_by_channel.items()
    )
    train_length = train_last + 1
    values = np.empty(
        (train_length + output_length * rolling_dates, channel_count), dtype=np.float32
    )
    for channel, entry in train_by_channel.items():
        target = np.asarray(entry["target"], dtype=np.float64)
        if target.size == 0 or not np.isfinite(target).all():
            raise ValueError(f"GP-Copula train series {channel} is empty or non-finite")
        offset = train_offsets[channel]
        values[:, channel] = np.float32(target.mean())
        values[offset : offset + target.size, channel] = target.astype(np.float32)

    dates: dict[int, list[int]] = {}
    assigned = np.zeros((output_length * rolling_dates, channel_count), dtype=bool)
    future = values[train_length:]
    for entry in test_entries:
        channel = _channel(entry=entry, channel_count=channel_count)
        start = _start(entry=entry)
        offset = _steps(start=start, first=first)
        target = np.asarray(entry["target"], dtype=np.float64)
        if target.size == 0 or not np.isfinite(target).all():
            raise ValueError(f"GP-Copula test series {channel} is empty or non-finite")
        end = offset + target.size - 1
        dates.setdefault(end, []).append(channel)

        train_entry = train_by_channel[channel]
        train_start = train_offsets[channel]
        overlap_start = max(offset, train_start)
        overlap_end = min(end, train_start + len(train_entry["target"]) - 1)
        if overlap_start <= overlap_end:
            raw_train = np.asarray(train_entry["target"], dtype=np.float64)
            actual = target[overlap_start - offset : overlap_end - offset + 1]
            reference = raw_train[
                overlap_start - train_start : overlap_end - train_start + 1
            ]
            if not np.array_equal(actual, reference):
                raise ValueError(
                    f"GP-Copula train/test overlap differs for channel {channel}"
                )

        begin = max(train_length, offset)
        finish = min(train_length + output_length * rolling_dates, end + 1)
        if begin < finish:
            source = target[begin - offset : finish - offset].astype(np.float32)
            destination = slice(begin - train_length, finish - train_length)
            occupied = assigned[destination, channel]
            existing = future[destination, channel]
            if occupied.any() and not np.array_equal(
                existing[occupied], source[occupied]
            ):
                raise ValueError(
                    f"GP-Copula rolling test values differ for channel {channel}"
                )
            existing[:] = source
            occupied[:] = True

    if len(dates) != rolling_dates or any(
        len(channels_at_date) != channel_count
        or set(channels_at_date) != set(range(channel_count))
        for channels_at_date in dates.values()
    ):
        raise ValueError("GP-Copula rolling dates do not cover every channel")
    end_steps = sorted(dates)
    if (
        any(
            right - left != output_length
            for left, right in zip(end_steps, end_steps[1:])
        )
        or end_steps[0] != train_last + output_length
    ):
        raise ValueError("GP-Copula test forecast dates do not follow the train fold")
    if not assigned.all():
        raise ValueError("GP-Copula test fold does not cover all forecast targets")

    timestamps = tuple(
        first + timedelta(hours=index) for index in range(values.shape[0])
    )
    frame = pl.DataFrame(
        {
            "date": timestamps,
            "fold": ["train"] * train_length
            + ["test"] * (output_length * rolling_dates),
            **{channel: values[:, index] for index, channel in enumerate(channels)},
        }
    )
    return frame, timestamps, train_length


def _channel(*, entry: dict[str, Any], channel_count: int) -> int:
    categories = entry.get("feat_static_cat")
    if (
        not isinstance(categories, list)
        or len(categories) != 1
        or type(categories[0]) is not int
        or not 0 <= categories[0] < channel_count
    ):
        raise ValueError("GP-Copula entry has an invalid static series ID")
    return categories[0]


def _start(*, entry: dict[str, Any]) -> datetime:
    try:
        return datetime.fromisoformat(entry["start"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("GP-Copula entry has an invalid start timestamp") from error


def _steps(*, start: datetime, first: datetime) -> int:
    hours = (start - first).total_seconds() / 3600
    if not hours.is_integer() or hours < 0:
        raise ValueError("GP-Copula timestamps are not aligned to the hourly grid")
    return int(hours)
