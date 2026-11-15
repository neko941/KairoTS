"""Google (GOOGL) daily stock prices, 2004 to 2019, the `stock` set of TimeGAN (Yoon
et al., 2019) and TSGBench.

One series of 3,685 trading days with six channels (Open, High, Low, Close,
Adj_Close, Volume). The CSV carries no dates, so the date is the trading-day index,
in the file's (chronological) order. With `returns`, every channel becomes its daily
log return `log x[t] - log x[t - 1]` and the first day is dropped (3,684 steps), the
form stylized-fact metrics read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class GoogleStock(Multivariate):
    TASKS = ("Forecasting", "Generation")
    URL = (
        "https://raw.githubusercontent.com/jsyoon0823/TimeGAN/"
        "b5070a32ce8cbb6b65e63acbef7842d67edd5458/data/stock_data.csv"
    )
    SHA256 = "134b8a00eb4eefa3242dd7ae85144082da80dedf75a1a656d5d08f97cd95277a"

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "GoogleStock"
        raw_path: str = "datasets/GoogleStock/raw"
        input_channels: tuple[str, ...] = (
            "Open",
            "High",
            "Low",
            "Close",
            "Adj_Close",
            "Volume",
        )
        returns: bool = False

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        return [Path(config.raw_path) / "stock_data.csv"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """TimeGAN's stock CSV on GitHub, pinned to a commit."""
        return (RemoteFile(path="stock_data.csv", urls=(cls.URL,), sha256=cls.SHA256),)

    @classmethod
    def load(cls, *, config: Config) -> GoogleStock:
        (path,) = cls.raw_files(config=config)
        if not path.is_file():
            raise FileNotFoundError(f"Google stock raw file does not exist: {path}")
        frame = pl.read_csv(path).cast(pl.Float64)
        missing = set(config.input_channels) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        frame = frame.select(config.input_channels)
        if config.returns:
            frame = frame.select(pl.all().log().diff()).slice(1)
        frame = frame.with_row_index(name="date").with_columns(
            pl.col("date").cast(pl.Int64)
        )
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={path.stem: tuple(frame.get_column("date").to_list())},
            clients=[(path.stem, frame.select(["date", *config.input_channels]))],
            manifest_fields={
                "raw_files": [str(path)],
                "frequency": "trading day",
                "returns": config.returns,
            },
        )
