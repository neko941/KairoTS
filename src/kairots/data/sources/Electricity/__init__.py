"""Electricity (UCI ElectricityLoadDiagrams20112014, processed), as TSLib's ECL.

Hourly consumption in kWh of 321 clients from 2016-07-01 to 2019-07-02 (26,304
steps): the processed form of the UCI load diagrams (15-minute kW readings of 370
clients) used by LSTNet and TSLib. The CSV is TSLib's copy; its channels are the
client indices `0` to `319` and `OT` (the last client). One file is one
multivariate client; `RowConcat` gives one univariate client per household.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class Electricity(Multivariate):
    TASKS = ("Forecasting",)
    URL = (
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"
        "electricity/electricity.csv"
    )
    SHA256 = "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad"

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "Electricity"
        raw_path: str = "datasets/Electricity/raw"
        input_channels: tuple[str, ...] = (*(str(index) for index in range(320)), "OT")

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        return [Path(config.raw_path) / "electricity.csv"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """TSLib's electricity CSV on Hugging Face."""
        return (RemoteFile(path="electricity.csv", urls=(cls.URL,), sha256=cls.SHA256),)

    @classmethod
    def load(cls, *, config: Config) -> Electricity:
        (path,) = cls.raw_files(config=config)
        if not path.is_file():
            raise FileNotFoundError(f"Electricity raw file does not exist: {path}")
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
