"""Beijing Multi-Site Air Quality (Zhang et al., 2017; UCI 501), a client per station.

Twelve national monitoring stations, hourly from 2013-03-01 to 2017-02-28 (35,064
rows each). The UCI archive holds a second zip with one CSV per station; its digest is
pinned and its members are read in memory by fixed name, never extracted. The 11
numeric channels are used; the categorical wind direction `wd` is dropped. Missing
values (`NA`) stay NaN, so the source needs `data.scaler.imputation = "masked"`:
statistics skip them and windows containing one are not formed.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Multivariate import Multivariate

URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
OUTER_SHA256 = "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"
INNER_NAME = "PRSA2017_Data_20130301-20170228.zip"
INNER_SHA256 = "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
STATIONS = (
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
)
ROWS = 35064


@dataclass(kw_only=True)
class BeijingAirQuality(Multivariate):
    TASKS = ("Forecasting",)
    REQUIRES = {"data.scaler.imputation": ("masked",)}

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "BeijingAirQuality"
        raw_path: str = "datasets/BeijingAirQuality/raw"
        input_channels: tuple[str, ...] = (
            "PM2.5",
            "PM10",
            "SO2",
            "NO2",
            "CO",
            "O3",
            "TEMP",
            "PRES",
            "DEWP",
            "RAIN",
            "WSPM",
        )
        seasonality: int = 24

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        return [Path(config.raw_path) / "archive" / INNER_NAME]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        return (
            RemoteFile(
                path="beijing.zip",
                urls=(URL,),
                sha256=OUTER_SHA256,
                extract="archive",
            ),
        )

    @classmethod
    def load(cls, *, config: Config) -> BeijingAirQuality:
        (path,) = cls.raw_files(config=config)
        if not path.is_file():
            raise FileNotFoundError(
                f"Beijing air quality raw file does not exist: {path}"
            )
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != INNER_SHA256:
            raise ValueError(f"{path.name} does not match its pinned sha256")
        clients: list[tuple[str, pl.DataFrame]] = []
        with zipfile.ZipFile(io.BytesIO(content)) as bundle:
            for station in STATIONS:
                text = bundle.read(
                    f"PRSA_Data_20130301-20170228/PRSA_Data_{station}_20130301-20170228.csv"
                )
                frame = pl.read_csv(
                    io.BytesIO(text), null_values="NA", infer_schema_length=None
                )
                if frame.height != ROWS:
                    raise ValueError(
                        f"{station}: expected {ROWS} rows, got {frame.height}"
                    )
                frame = frame.select(
                    pl.datetime(
                        pl.col("year"), pl.col("month"), pl.col("day"), pl.col("hour")
                    ).alias("date"),
                    *(
                        pl.col(name).cast(pl.Float32).fill_null(float("nan"))
                        for name in config.input_channels
                    ),
                )
                clients.append((station, frame))
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={
                name: tuple(frame["date"].to_list()) for name, frame in clients
            },
            clients=clients,
            manifest_fields={"raw_files": [str(path)], "frequency": "1h"},
        )
