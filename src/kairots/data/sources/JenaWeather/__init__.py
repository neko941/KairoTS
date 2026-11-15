"""Jena weather (Max Planck Institute for Biogeochemistry, 2020), as TSLib's Weather.

One rooftop station in Jena, 21 meteorological channels every 10 minutes for 2020
(52,696 steps). The CSV is TSLib's copy; its headers carry units (`p (mbar)`), so
channels are renamed to the part before the unit (`p`, `T`, `max_wv`, ...).
Like TSLib, the station's -9999 error codes in the wind channels are kept as values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class JenaWeather(Multivariate):
    TASKS = ("Forecasting",)
    URL = (
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"
        "weather/weather.csv"
    )
    SHA256 = "34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63"

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "JenaWeather"
        raw_path: str = "datasets/JenaWeather/raw"
        input_channels: tuple[str, ...] = (
            "p",
            "T",
            "Tpot",
            "Tdew",
            "rh",
            "VPmax",
            "VPact",
            "VPdef",
            "sh",
            "H2OC",
            "rho",
            "wv",
            "max_wv",
            "wd",
            "rain",
            "raining",
            "SWDR",
            "PAR",
            "max_PAR",
            "Tlog",
            "OT",
        )

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        return [Path(config.raw_path) / "weather.csv"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """TSLib's weather CSV on Hugging Face."""
        return (RemoteFile(path="weather.csv", urls=(cls.URL,), sha256=cls.SHA256),)

    @classmethod
    def load(cls, *, config: Config) -> JenaWeather:
        (path,) = cls.raw_files(config=config)
        if not path.is_file():
            raise FileNotFoundError(f"Jena weather raw file does not exist: {path}")
        # Unit headers hold Latin-1 bytes (`W/m²`); they are dropped by the rename.
        frame = pl.read_csv(path, encoding="utf8-lossy", try_parse_dates=True)
        frame = frame.rename(
            {name: name.split(" (")[0].replace(". ", "_") for name in frame.columns}
        )
        # 2020-05-12 06:00 appears twice; a repeated date would make the window step
        # zero. The 1 h 40 min gap on 2020-05-29 stays a gap that windows skip.
        frame = frame.unique(subset="date", keep="first", maintain_order=True)
        missing = {"date", *config.input_channels} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={path.stem: tuple(frame.get_column("date").to_list())},
            clients=[(path.stem, frame.select(["date", *config.input_channels]))],
            manifest_fields={"raw_files": [str(path)], "frequency": "10min"},
        )
