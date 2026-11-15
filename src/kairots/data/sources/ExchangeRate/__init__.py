"""TSLib/LSTNet daily exchange rates of eight currencies, 7,588 rows (1990-2010)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class ExchangeRate(Multivariate):
    TASKS = ("Forecasting", "SampleForecasting")
    URL = (
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"
        "exchange_rate/exchange_rate.csv"
    )
    SHA256 = "48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842"

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "ExchangeRate"
        raw_path: str = "datasets/ExchangeRate/raw"
        input_channels: tuple[str, ...] = (*(str(index) for index in range(7)), "OT")

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        return [Path(config.raw_path) / "exchange_rate.csv"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """TSLib's exchange_rate CSV on Hugging Face."""
        return (
            RemoteFile(path="exchange_rate.csv", urls=(cls.URL,), sha256=cls.SHA256),
        )

    @classmethod
    def load(cls, *, config: Config) -> ExchangeRate:
        (path,) = cls.raw_files(config=config)
        if not path.is_file():
            raise FileNotFoundError(f"ExchangeRate raw file does not exist: {path}")
        frame = pl.read_csv(path, try_parse_dates=True)
        missing = {"date", *config.input_channels} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={path.stem: tuple(frame.get_column("date").to_list())},
            clients=[(path.stem, frame.select(["date", *config.input_channels]))],
            manifest_fields={"raw_files": [str(path)], "frequency": "daily"},
        )
