"""CDC national illness (ILI) weekly indicators, as TSLib's national_illness.

Seven weekly series from 2002 to 2020, 966 steps: weighted and unweighted ILI
percentages, two age bands, total ILI counts, provider counts and OT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class ILI(Multivariate):
    TASKS = ("Forecasting",)
    URL = (
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"
        "illness/national_illness.csv"
    )
    SHA256 = "93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a"

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "ILI"
        raw_path: str = "datasets/ILI/raw"
        input_channels: tuple[str, ...] = (
            "% WEIGHTED ILI",
            "%UNWEIGHTED ILI",
            "AGE 0-4",
            "AGE 5-24",
            "ILITOTAL",
            "NUM. OF PROVIDERS",
            "OT",
        )
        seasonality: int = 52

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        return [Path(config.raw_path) / "national_illness.csv"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """TSLib's national_illness CSV on Hugging Face."""
        return (
            RemoteFile(path="national_illness.csv", urls=(cls.URL,), sha256=cls.SHA256),
        )

    @classmethod
    def load(cls, *, config: Config) -> ILI:
        (path,) = cls.raw_files(config=config)
        if not path.is_file():
            raise FileNotFoundError(f"ILI raw file does not exist: {path}")
        frame = pl.read_csv(path, try_parse_dates=True)
        missing = {"date", *config.input_channels} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={path.stem: tuple(frame.get_column("date").to_list())},
            clients=[(path.stem, frame.select(["date", *config.input_channels]))],
            manifest_fields={"raw_files": [str(path)], "frequency": "weekly"},
        )
