"""The M3 competition's four-frequency collection sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import xlrd

from kairots.data.download import RemoteFile
from kairots.data.sources.Collection import Collection

URL = (
    "https://raw.githubusercontent.com/jordicolomer/m3-competition/"
    "5364d59beb5ec7433bbc8188b1db54b0b7662974/M3C.xls"
)
SHA256 = "23bfbba215dad49eb9b0f315a3e38bcc541f78f4689b168616ef25a5e8f293a6"


@dataclass(kw_only=True)
class M3(Collection):
    ABSTRACT = True
    FREQUENCY: ClassVar[str] = ""
    SHEET: ClassVar[str] = ""
    SERIES_COUNT: ClassVar[int] = 0
    HORIZON: ClassVar[int] = 0
    START_FIELDS: ClassVar[tuple[str, ...]] = ()

    @dataclass(kw_only=True)
    class Config(Collection.Config):
        name: str = "M3"
        raw_path: str = "datasets/M3/raw"

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        return [Path(config.raw_path) / "M3C.xls"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        return (RemoteFile(path="M3C.xls", urls=(URL,), sha256=SHA256),)

    @classmethod
    def load(cls, *, config: Config) -> M3:
        path = cls.raw_files(config=config)[0]
        workbook = xlrd.open_workbook(str(path), on_demand=True)
        try:
            sheet = workbook.sheet_by_name(cls.SHEET)
            clients: list[tuple[str, pl.DataFrame]] = []
            series_metadata: list[dict[str, str | int]] = []
            identifiers: set[str] = set()
            for row_index in range(1, sheet.nrows):
                raw_id = str(sheet.cell_value(row_index, 0)).strip()
                series_id = "".join(raw_id.split())
                if not series_id or series_id in identifiers:
                    raise ValueError(f"duplicate or empty M3 series ID: {raw_id!r}")
                identifiers.add(series_id)

                total_length = int(sheet.cell_value(row_index, 1))
                forecast_length = int(sheet.cell_value(row_index, 2))
                if (
                    forecast_length != cls.HORIZON
                    or not 0 < forecast_length < total_length
                ):
                    raise ValueError(
                        f"{series_id} has invalid M3 horizon/length: "
                        f"{forecast_length}/{total_length}, "
                        f"expected horizon {cls.HORIZON}"
                    )

                data_cells = sheet.row_values(row_index, 6)
                last_value = max(
                    (
                        index
                        for index, value in enumerate(data_cells)
                        if value not in ("", None)
                    ),
                    default=-1,
                )
                if last_value + 1 != total_length or any(
                    value in ("", None) for value in data_cells[: last_value + 1]
                ):
                    raise ValueError(
                        f"{series_id} has {last_value + 1} values; "
                        f"M3 declares {total_length} contiguous observations"
                    )
                values = np.asarray(data_cells[:total_length], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"{series_id} contains a non-finite value")

                frame = pl.DataFrame(
                    {
                        "date": np.arange(total_length),
                        "fold": ["train"] * (total_length - forecast_length)
                        + ["test"] * forecast_length,
                        "value": values,
                    }
                )
                clients.append((series_id, frame))
                metadata: dict[str, str | int] = {
                    "series_id": series_id,
                    "category": str(sheet.cell_value(row_index, 3)).strip(),
                }
                for column_index, field in enumerate(cls.START_FIELDS, start=4):
                    metadata[field] = int(sheet.cell_value(row_index, column_index))
                series_metadata.append(metadata)

            if len(clients) != cls.SERIES_COUNT:
                raise ValueError(
                    f"{cls.SHEET} contains {len(clients)} rows; "
                    f"expected {cls.SERIES_COUNT}"
                )
            return cls(
                name=cls.__name__,
                input_channels=config.input_channels,
                timestamps={},
                clients=clients,
                manifest_fields={
                    "frequency": cls.FREQUENCY,
                    "forecast_horizon": cls.HORIZON,
                    "series": series_metadata,
                    "raw_files": [str(path)],
                },
            )
        finally:
            workbook.release_resources()
