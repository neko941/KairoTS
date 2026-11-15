"""Feature join: the same timeline observed by different sources, as one client.

Each member source has one client. Its declared channels are renamed
`<member>/<channel>` and the members are inner-joined on exact `date` matches, so
the result holds only the timestamps every member has (no fill, no nearest-time
matching, which could read a later observation). A duplicate `date` in a member
raises, since its join partner would be ambiguous. Channel names must be declared
in `input_channels`, because the model's width is set from the config before any
data is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any

import polars as pl

from kairots.data.sources.Composite import Composite
from kairots.registry import get


@dataclass(kw_only=True)
class FeatureJoin(Composite):
    TASKS = ("Forecasting",)

    @dataclass(kw_only=True)
    class Config(Composite.Config):
        name: str = "FeatureJoin"
        members: tuple[str, ...] = ("ETTh1", "Electricity")
        # One raw directory per member, in the order of `members`.
        raw_paths: tuple[str, ...] = (
            "datasets/ETDataset/raw",
            "datasets/Electricity/raw",
        )
        input_channels: tuple[str, ...] = ("ETTh1/OT", "Electricity/OT")

    @staticmethod
    def _members(*, config: Config) -> list[tuple[str, Any, Any]]:
        if len(config.members) != len(config.raw_paths):
            raise ValueError("members and raw_paths differ in length")
        members = []
        for name, raw_path in zip(config.members, config.raw_paths, strict=True):
            channels = tuple(
                channel.removeprefix(f"{name}/")
                for channel in config.input_channels
                if channel.startswith(f"{name}/")
            )
            if not channels:
                raise ValueError(f"input_channels name no channel of {name}")
            member_type = get(category="data/sources", name=name)
            members.append(
                (
                    name,
                    member_type,
                    member_type.Config(raw_path=raw_path, input_channels=channels),
                )
            )
        return members

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        return [
            path
            for _, member_type, member_config in cls._members(config=config)
            for path in member_type.raw_files(config=member_config)
        ]

    @classmethod
    def download(cls, *, config: Config) -> None:
        for _, member_type, member_config in cls._members(config=config):
            member_type.download(config=member_config)

    @classmethod
    def load(cls, *, config: Config) -> FeatureJoin:
        frames = []
        for name, member_type, member_config in cls._members(config=config):
            member = member_type.load(config=member_config)
            if len(member.clients) != 1:
                raise ValueError(f"{name} has {len(member.clients)} clients, needs 1")
            ((_, frame),) = member.clients
            if frame.get_column("date").is_duplicated().any():
                raise ValueError(f"{name} has duplicate dates")
            frames.append(
                frame.select(
                    "date",
                    *[
                        pl.col(channel).alias(f"{name}/{channel}")
                        for channel in member_config.input_channels
                    ],
                )
            )
        joined = reduce(
            lambda left, right: left.join(right, on="date", how="inner"), frames
        ).sort("date")
        columns = [column for column in joined.columns if column != "date"]
        if columns != list(config.input_channels):
            raise ValueError(
                f"joined columns {columns} differ from input_channels "
                f"{list(config.input_channels)}"
            )
        if joined.is_empty():
            raise ValueError(f"{config.members} share no dates")
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={cls.__name__: tuple(joined.get_column("date").to_list())},
            clients=[(cls.__name__, joined)],
            manifest_fields={
                "raw_files": [str(path) for path in cls.raw_files(config=config)],
                "members": list(config.members),
            },
        )
