"""Contracts shared by dataset sources."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from kairots.data.download import RemoteFile, fetch


@dataclass(kw_only=True)
class Source:
    ABSTRACT = True

    @dataclass(kw_only=True)
    class Config:
        name: str = ""
        raw_path: str = ""
        # Fetch missing raw files from the source's URLs before preparing.
        download: bool = True
        input_channels: tuple[str, ...] = ()
        batch_size: int = 32
        # Season length in steps: the naive lag MASE scales by (TSLib frequency_map).
        seasonality: int = 1

    name: str
    input_channels: tuple[str, ...]
    timestamps: dict[str, tuple[Any, ...]]
    clients: list[tuple[str, pl.DataFrame]]
    manifest_fields: dict[str, Any]

    @classmethod
    def raw_files(cls, *, config: Any) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        raise NotImplementedError

    @classmethod
    def remote_files(cls, *, config: Any) -> tuple[RemoteFile, ...]:
        """Where each raw file comes from; empty when files are placed by hand."""
        return ()

    @classmethod
    def download(cls, *, config: Any) -> None:
        """Fetch missing raw files from `remote_files`, then check none are missing."""
        for remote in cls.remote_files(config=config):
            fetch(remote=remote, root=Path(config.raw_path))
        missing = [path for path in cls.raw_files(config=config) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{cls.__name__} raw files are missing and have no download: {missing}"
            )


class SingleMember:
    @dataclass(kw_only=True)
    class Config(Source.Config):
        name: str = "SingleMember"
        member: str = ""
