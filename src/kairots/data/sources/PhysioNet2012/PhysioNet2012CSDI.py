"""The CSDI paper's 48-hour, 35-variable Set-A imputation cohort."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Panel import Panel
from kairots.data.sources.PhysioNet2012 import (
    CSDI_VARIABLES,
    URL_ROOT,
    PhysioNet2012,
)


@dataclass(kw_only=True)
class PhysioNet2012CSDI(Panel):
    TASKS = ("ProbabilisticImputation",)
    SHA256 = {"set-a.tar.gz": PhysioNet2012.SHA256["set-a.tar.gz"]}

    @dataclass(kw_only=True)
    class Config(Panel.Config):
        name: str = "PhysioNet2012CSDI"
        raw_path: str = "datasets/PhysioNet2012/raw"
        input_length: int = 48
        input_channels: tuple[str, ...] = CSDI_VARIABLES
        num_classes: int = 2
        normalization: str = "historical"

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        return [Path(config.raw_path) / name for name in cls.SHA256]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        return tuple(
            RemoteFile(path=name, urls=(f"{URL_ROOT}/{name}",), sha256=sha256)
            for name, sha256 in cls.SHA256.items()
        )

    @classmethod
    def load(cls, *, config: Config) -> PhysioNet2012CSDI:
        """Aggregate the last observed value in each patient/hour/variable cell."""
        rows = PhysioNet2012.read_set(path=cls.raw_files(config=config)[0])
        patients = rows.get_column("record").unique().sort().to_list()
        rows = rows.with_columns(
            pl.col("Time").str.split(":").list.get(0).cast(pl.Int32).alias("hour")
        ).filter(pl.col("Parameter").is_in(config.input_channels))
        wide = (
            rows.group_by("record", "hour", "Parameter", maintain_order=True)
            .agg(pl.col("Value").last())
            .pivot(on="Parameter", index=["record", "hour"], values="Value")
        )
        wide = wide.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(name)
            for name in config.input_channels
            if name not in wide.columns
        )
        samples = pl.DataFrame({"record": patients}).with_row_index("sample")
        grid = (
            samples.join(
                pl.DataFrame({"step": range(config.input_length)}), how="cross"
            )
            .join(
                wide,
                left_on=["record", "step"],
                right_on=["record", "hour"],
                how="left",
            )
            .with_columns(
                pl.col("sample", "step").cast(pl.Int32),
                pl.lit("train").alias("fold"),
                pl.lit(0, dtype=pl.Int32).alias("label"),
            )
            .select("sample", "step", "fold", "label", *config.input_channels)
            .sort("sample", "step")
        )
        if config.normalization == "historical":
            stats = grid.select(
                pl.col(name).mean().alias(f"{name}_mean")
                for name in config.input_channels
            ).row(0, named=True)
            scales = grid.select(
                pl.col(name).std(ddof=0).alias(f"{name}_scale")
                for name in config.input_channels
            ).row(0, named=True)
            grid = grid.with_columns(
                (
                    (pl.col(name) - stats[f"{name}_mean"])
                    / max(float(scales[f"{name}_scale"] or 0), 1e-12)
                ).alias(name)
                for name in config.input_channels
            )
        elif config.normalization != "train_only":
            raise ValueError("normalization must be 'historical' or 'train_only'")
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={},
            clients=[("PhysioNet2012", grid)],
            manifest_fields={
                "raw_files": [str(path) for path in cls.raw_files(config=config)],
                "normalization": config.normalization,
            },
        )
