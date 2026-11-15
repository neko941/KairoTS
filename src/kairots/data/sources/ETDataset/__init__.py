"""Shared schema and CSV reader for the ETT benchmark family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class ETDataset(Multivariate):
    ABSTRACT = True
    FILE_NAME: ClassVar[str] = ""
    FREQUENCY: ClassVar[str] = ""
    SHA256: ClassVar[str] = ""
    URL_ROOT = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "ETDataset"
        raw_path: str = "datasets/ETDataset/raw"
        input_channels: tuple[str, ...] = (
            "HUFL",
            "HULL",
            "MUFL",
            "MULL",
            "LUFL",
            "LULL",
            "OT",
        )

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        return [Path(config.raw_path) / cls.FILE_NAME / f"{cls.FILE_NAME}.csv"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """The variant's CSV from the official ETDataset repository."""
        return (
            RemoteFile(
                path=f"{cls.FILE_NAME}/{cls.FILE_NAME}.csv",
                urls=(f"{cls.URL_ROOT}/{cls.FILE_NAME}.csv",),
                sha256=cls.SHA256,
            ),
        )

    @classmethod
    def load(cls, *, config: Config) -> ETDataset:
        (path,) = cls.raw_files(config=config)
        if not path.is_file():
            raise FileNotFoundError(f"ETT raw file does not exist: {path}")
        frame = pl.read_csv(path, try_parse_dates=True)
        required = {"date", *config.input_channels}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={path.stem: tuple(frame.get_column("date").to_list())},
            clients=[(path.stem, frame.select(["date", *config.input_channels]))],
            manifest_fields={"raw_files": [str(path)], "frequency": cls.FREQUENCY},
        )
