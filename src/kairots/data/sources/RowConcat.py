"""Row concatenation: every channel of a member source becomes a univariate series.

A multivariate client of C channels becomes C univariate clients, named
`<member>/<client>/<channel>`, each with the member's dates and one channel
`value`. `Centralized` then trains one univariate model on all of them stacked
by rows (as for a household-per-channel set such as Electricity); a federated
strategy treats each series as a client. Each series keeps its own cuts and its
own train-fitted scaler. `ColumnConcat` is the inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from kairots.data.sources.base import SingleMember
from kairots.data.sources.Composite import Composite
from kairots.registry import get


@dataclass(kw_only=True)
class RowConcat(Composite):
    TASKS = ("Forecasting",)

    @dataclass(kw_only=True)
    class Config(SingleMember.Config):
        name: str = "RowConcat"
        member: str = "JenaWeather"
        # The member's raw directory.
        raw_path: str = "datasets/JenaWeather/raw"
        input_channels: tuple[str, ...] = ("value",)
        # Member channels to split into series; empty splits all of them.
        channels: tuple[str, ...] = ()

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
    def load(cls, *, config: Config) -> RowConcat:
        member_type, member_config = cls._member(config=config)
        member = member_type.load(config=member_config)
        channels = tuple(config.channels) or member.input_channels
        unknown = set(channels) - set(member.input_channels)
        if unknown:
            raise ValueError(f"{member.name} has no channels {sorted(unknown)}")
        clients = [
            (
                f"{member.name}/{name}/{channel}",
                frame.select("date", pl.col(channel).alias("value")),
            )
            for name, frame in member.clients
            for channel in channels
        ]
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={
                name: tuple(frame.get_column("date").to_list())
                for name, frame in clients
            },
            clients=clients,
            manifest_fields={
                "raw_files": member.manifest_fields["raw_files"],
                "member": member.name,
                "frequency": member.manifest_fields.get("frequency"),
            },
        )
