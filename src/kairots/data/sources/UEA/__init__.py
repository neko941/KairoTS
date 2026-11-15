"""UEA multivariate classification archive: shared `.ts` reader and mirrors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Panel import Panel

# The archive site (timeseriesclassification.com) answers 401 to scripts; aeon and
# sktime ship byte-identical copies of the small problems.
MIRRORS = (
    "https://raw.githubusercontent.com/aeon-toolkit/aeon/main/aeon/datasets/data",
    "https://raw.githubusercontent.com/sktime/sktime/main/sktime/datasets/data",
)
FOLDS = ("TRAIN", "TEST")


@dataclass(kw_only=True)
class UEA(Panel):
    ABSTRACT = True
    # Per variant: problem name and the sha256 of its TRAIN and TEST files.
    PROBLEM: ClassVar[str] = ""
    SHA256: ClassVar[dict[str, str]] = {}

    @dataclass(kw_only=True)
    class Config(Panel.Config):
        name: str = "UEA"
        raw_path: str = "datasets/UEA/raw"

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        return [
            Path(config.raw_path) / cls.PROBLEM / f"{cls.PROBLEM}_{fold}.ts"
            for fold in FOLDS
        ]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        return tuple(
            RemoteFile(
                path=f"{cls.PROBLEM}/{cls.PROBLEM}_{fold}.ts",
                urls=tuple(
                    f"{mirror}/{cls.PROBLEM}/{cls.PROBLEM}_{fold}.ts"
                    for mirror in MIRRORS
                ),
                sha256=cls.SHA256[fold],
            )
            for fold in FOLDS
        )

    @staticmethod
    def read_ts(*, path: Path) -> tuple[list[str], list[list[list[float]]], list[str]]:
        """Parse an equal-length `.ts` file into (classes, samples, labels).

        Each data line is `dim_0 values : dim_1 values : ... : label`, values
        comma-separated; `samples[i][d]` is dimension d of sample i.
        """
        classes: list[str] = []
        samples: list[list[list[float]]] = []
        labels: list[str] = []
        in_data = False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                header = line.lower()
                if header.startswith("@classlabel"):
                    classes = line.split()[2:]
                elif header.startswith("@equallength") and line.split()[1] != "true":
                    raise ValueError(f"{path.name}: unequal lengths are not supported")
                elif header == "@data":
                    in_data = True
                continue
            *dimensions, label = line.split(":")
            samples.append(
                [[float(value) for value in part.split(",")] for part in dimensions]
            )
            labels.append(label)
        return classes, samples, labels

    @classmethod
    def load(cls, *, config: Config) -> UEA:
        frames: list[pl.DataFrame] = []
        classes: list[str] = []
        sample_id = 0
        for fold, path in zip(FOLDS, cls.raw_files(config=config), strict=True):
            classes, samples, labels = cls.read_ts(path=path)
            for dimensions, label in zip(samples, labels, strict=True):
                if len(dimensions) != len(config.input_channels):
                    raise ValueError(
                        f"{path.name}: {len(dimensions)} dimensions, config names "
                        f"{len(config.input_channels)} channels"
                    )
                length = len(dimensions[0])
                if length != config.input_length:
                    raise ValueError(
                        f"{path.name}: length {length}, config input_length "
                        f"{config.input_length}"
                    )
                frames.append(
                    pl.DataFrame(
                        {
                            "sample": [sample_id] * length,
                            "step": list(range(length)),
                            "fold": [fold.lower()] * length,
                            "label": [classes.index(label)] * length,
                            **dict(zip(config.input_channels, dimensions, strict=True)),
                        },
                        schema_overrides={
                            "sample": pl.Int32,
                            "step": pl.Int32,
                            "label": pl.Int32,
                        },
                    )
                )
                sample_id += 1
        if len(classes) != config.num_classes:
            raise ValueError(
                f"{cls.PROBLEM}: {len(classes)} classes, config num_classes "
                f"{config.num_classes}"
            )
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={},
            clients=[(cls.PROBLEM, pl.concat(frames))],
            manifest_fields={
                "raw_files": [str(path) for path in cls.raw_files(config=config)],
                "classes": classes,
            },
        )
