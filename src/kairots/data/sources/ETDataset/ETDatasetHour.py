"""Hourly ETT composite, ordered ETTh1 then ETTh2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kairots.data.download import RemoteFile
from kairots.data.sources.Composite import Composite
from kairots.data.sources.ETDataset import ETDataset
from kairots.data.sources.ETDataset.ETTh1 import ETTh1
from kairots.data.sources.ETDataset.ETTh2 import ETTh2


class ETDatasetHour(Composite):
    @dataclass(kw_only=True)
    class Config(ETDataset.Config):
        name: str = "ETDatasetHour"

    MEMBERS = (ETTh1, ETTh2)

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files of every member, in member order."""
        return [
            path
            for member in cls.MEMBERS
            for path in member.raw_files(
                config=member.Config(
                    raw_path=config.raw_path, input_channels=config.input_channels
                )
            )
        ]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """Every member's files; members share this source's raw directory."""
        return tuple(
            remote
            for member in cls.MEMBERS
            for remote in member.remote_files(config=member.Config())
        )

    @classmethod
    def load(cls, *, config: Config) -> ETDatasetHour:
        members = [
            member.load(
                config=member.Config(
                    raw_path=config.raw_path, input_channels=config.input_channels
                )
            )
            for member in cls.MEMBERS
        ]
        clients = [(name, frame) for _, name, frame in Composite.join(members=members)]
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={
                key: value
                for member in members
                for key, value in member.timestamps.items()
            },
            clients=clients,
            manifest_fields={
                "raw_files": [
                    path
                    for member in members
                    for path in member.manifest_fields["raw_files"]
                ],
                "members": [member.name for member in members],
                "frequency": "hourly",
            },
        )
