"""Traffic (Caltrans PeMS road occupancy), as TSLib's Traffic.

Hourly road occupancy rates from 862 sensors over 17,544 steps (2016-2018),
the multivariate benchmark of LTSF-Linear and TSLib. The CSV is TSLib's copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class Traffic(Multivariate):
    TASKS = ("Forecasting",)
    URL = (
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"
        "traffic/traffic.csv"
    )
    SHA256 = "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "Traffic"
        raw_path: str = "datasets/Traffic/raw"
        input_channels: tuple[str, ...] = (
            *(str(index) for index in range(861)),
            "OT",
        )

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        return [Path(config.raw_path) / "traffic.csv"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """TSLib's traffic CSV on Hugging Face."""
        return (RemoteFile(path="traffic.csv", urls=(cls.URL,), sha256=cls.SHA256),)

    @classmethod
    def load(cls, *, config: Config) -> Traffic:
        (path,) = cls.raw_files(config=config)
        if not path.is_file():
            raise FileNotFoundError(f"Traffic raw file does not exist: {path}")
        frame = pl.read_csv(path, try_parse_dates=True)
        missing = {"date", *config.input_channels} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={path.stem: tuple(frame.get_column("date").to_list())},
            clients=[(path.stem, frame.select(["date", *config.input_channels]))],
            manifest_fields={"raw_files": [str(path)], "frequency": "hourly"},
        )
