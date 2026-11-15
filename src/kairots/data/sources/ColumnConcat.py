"""Column concatenation: every client of a member source becomes channels of one series.

Each member client's channels are renamed `<client>/<channel>` and the clients
are inner-joined on `date`, so the result is one multivariate client that holds
only the dates every member client has (no fill). Channel names must be declared
in `input_channels`, because the model's width is set from the config before any
data is read. `RowConcat` is the inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any

import polars as pl

from kairots.data.sources.base import SingleMember
from kairots.data.sources.Composite import Composite
from kairots.registry import get

STATIONS = ("01300099999", "02490099999", "02498099999")


@dataclass(kw_only=True)
class ColumnConcat(Composite):
    TASKS = ("Forecasting",)

    @dataclass(kw_only=True)
    class Config(SingleMember.Config):
        name: str = "ColumnConcat"
        member: str = "TinyWeather5K"
        # The member's raw directory.
        raw_path: str = "datasets/TinyWeather5K/raw"
        input_channels: tuple[str, ...] = tuple(
            f"{station}/{channel}"
            for station in STATIONS
            for channel in ("TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP")
        )

    @staticmethod
    def _member(*, config: Config) -> tuple[Any, Any]:
        member_type = get(category="data/sources", name=config.member)
        return member_type, member_type.Config(raw_path=config.raw_path)

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        member_type, member_config = cls._member(config=config)
        return member_type.raw_files(config=member_config)

    @classmethod
    def download(cls, *, config: Config) -> None:
        member_type, member_config = cls._member(config=config)
        member_type.download(config=member_config)

    @classmethod
    def load(cls, *, config: Config) -> ColumnConcat:
        member_type, member_config = cls._member(config=config)
        member = member_type.load(config=member_config)
        frames = [
            frame.select(
                "date",
                *[
                    pl.col(channel).alias(f"{name}/{channel}")
                    for channel in member.input_channels
                ],
            )
            for name, frame in member.clients
        ]
        joined = reduce(
            lambda left, right: left.join(right, on="date", how="inner"), frames
        ).sort("date")
        columns = [column for column in joined.columns if column != "date"]
        if columns != list(config.input_channels):
            raise ValueError(
                f"{member.name} columns {columns} differ from input_channels "
                f"{list(config.input_channels)}"
            )
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={member.name: tuple(joined.get_column("date").to_list())},
            clients=[(member.name, joined)],
            manifest_fields={
                "raw_files": member.manifest_fields["raw_files"],
                "member": member.name,
                "frequency": member.manifest_fields.get("frequency"),
            },
        )
