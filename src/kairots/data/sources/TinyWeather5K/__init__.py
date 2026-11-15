"""TinyWeather5K: a subset of WEATHER-5K stations (Han et al., 2024).

Each station CSV holds hourly observations from 2014 to 2023 (87,648 steps) of
temperature (`TMP`, °C), dew point (`DEW`, °C), wind direction (`WND_ANGLE`, °),
wind speed (`WND_RATE`, m/s) and sea-level pressure (`SLP`, hPa). WEATHER-5K
fills unobserved steps; its `MASK` column, which flags them, is not read. There
is no download: station files are placed by hand under `raw_path`
(https://github.com/taohan10200/WEATHER-5K). One station is one client.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class TinyWeather5K(Multivariate):
    TASKS = ("Forecasting",)

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "TinyWeather5K"
        raw_path: str = "datasets/TinyWeather5K/raw"
        input_channels: tuple[str, ...] = ("TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP")
        # Station file stems; empty reads every CSV under raw_path.
        stations: tuple[str, ...] = ()

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        root = Path(config.raw_path)
        if config.stations:
            return [root / f"{station}.csv" for station in config.stations]
        return sorted(root.glob("*.csv"))

    @classmethod
    def load(cls, *, config: Config) -> TinyWeather5K:
        paths = cls.raw_files(config=config)
        if not paths:
            raise FileNotFoundError(
                f"no WEATHER-5K station files in {config.raw_path}; download station "
                "CSVs from https://github.com/taohan10200/WEATHER-5K and place them "
                "there, one <station>.csv per client"
            )
        clients = []
        for path in paths:
            frame = pl.read_csv(path, try_parse_dates=True).rename({"DATE": "date"})
            missing = {"date", *config.input_channels} - set(frame.columns)
            if missing:
                raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
            clients.append((path.stem, frame.select(["date", *config.input_channels])))
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={
                name: tuple(frame.get_column("date").to_list())
                for name, frame in clients
            },
            clients=clients,
            manifest_fields={
                "raw_files": [str(path) for path in paths],
                "frequency": "hourly",
            },
        )
