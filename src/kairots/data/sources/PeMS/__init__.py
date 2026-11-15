"""PeMS traffic sensor network datasets."""

from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.SpatioTemporal import SpatioTemporal

URL_ROOT = "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/refs/heads/main/data"


@dataclass(kw_only=True)
class PeMS(SpatioTemporal):
    ABSTRACT = True
    DISTRICT: ClassVar[str] = ""
    START: ClassVar[datetime]
    SHA256: ClassVar[dict[str, str]] = {}

    @dataclass(kw_only=True)
    class Config(SpatioTemporal.Config):
        name: str = "PeMS"
        raw_path: str = "datasets/PeMS/raw"
        input_channels: tuple[str, ...] = ("flow", "occupancy", "speed")

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        root = Path(config.raw_path) / cls.DISTRICT
        return [root / f"{cls.DISTRICT}.{suffix}" for suffix in ("npz", "csv")]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        return tuple(
            RemoteFile(
                path=f"{cls.DISTRICT}/{cls.DISTRICT}.{suffix}",
                urls=(f"{URL_ROOT}/{cls.DISTRICT}/{cls.DISTRICT}.{suffix}",),
                sha256=cls.SHA256[suffix],
            )
            for suffix in ("npz", "csv")
        )

    @classmethod
    def load(cls, *, config: Config) -> PeMS:
        npz_path, csv_path = cls.raw_files(config=config)
        with np.load(npz_path) as archive:
            values = archive["data"]
        if values.ndim != 3 or values.shape[-1] != len(config.input_channels):
            raise ValueError("PeMS array feature axis does not match input_channels")
        steps, num_nodes, _ = values.shape
        network_values = values.transpose(1, 0, 2).reshape(
            -1, len(config.input_channels)
        )
        dates = np.tile(
            np.datetime64(cls.START, "us") + np.arange(steps) * np.timedelta64(5, "m"),
            num_nodes,
        )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            graph = [
                [int(edge["from"]), int(edge["to"]), float(edge["cost"])]
                for edge in DictReader(handle)
            ]
        frame = pl.DataFrame(
            {
                "date": dates,
                "node": np.repeat(np.arange(num_nodes, dtype=np.int32), steps),
                **dict(zip(config.input_channels, network_values.T, strict=True)),
            }
        )
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={},
            clients=[(cls.DISTRICT, frame)],
            manifest_fields={
                "raw_files": [str(path) for path in (npz_path, csv_path)],
                "num_nodes": num_nodes,
                "graph": graph,
            },
        )
