"""Server Machine Dataset (Su et al., KDD 2019, OmniAnomaly): one client per machine.

Each machine has a train file (unlabeled, mostly normal), a test file and per-step
test labels, 38 channels each. Rows get an integer `date` (train then test), a
`fold` (`train`/`test`) and a `label` (0 on train: anomaly detection here trains on
normal-only data and never reads train labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Multivariate import Multivariate

URL_ROOT = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/"
    "ServerMachineDataset"
)
PARTS = ("train", "test", "test_label")


@dataclass(kw_only=True)
class SMD(Multivariate):
    ABSTRACT = True
    # Per variant: machine -> sha256 of its train, test and test_label files.
    MACHINES: ClassVar[dict[str, tuple[str, str, str]]] = {}

    @dataclass(kw_only=True)
    class Config(Multivariate.Config):
        name: str = "SMD"
        raw_path: str = "datasets/SMD/raw"
        input_channels: tuple[str, ...] = tuple(
            f"channel_{index:02d}" for index in range(38)
        )

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        return [
            Path(config.raw_path) / part / f"{machine}.txt"
            for machine in cls.MACHINES
            for part in PARTS
        ]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        return tuple(
            RemoteFile(
                path=f"{part}/{machine}.txt",
                urls=(f"{URL_ROOT}/{part}/{machine}.txt",),
                sha256=digest,
            )
            for machine, digests in cls.MACHINES.items()
            for part, digest in zip(PARTS, digests, strict=True)
        )

    @classmethod
    def load(cls, *, config: Config) -> SMD:
        root = Path(config.raw_path)
        clients: list[tuple[str, pl.DataFrame]] = []
        for machine in cls.MACHINES:
            train, test, labels = (
                np.loadtxt(root / part / f"{machine}.txt", delimiter=",", ndmin=2)
                for part in PARTS
            )
            if train.shape[1] != len(config.input_channels) or len(labels) != len(test):
                raise ValueError(f"{machine}: unexpected SMD file shapes")
            values = np.concatenate([train, test]).astype(np.float32)
            clients.append(
                (
                    machine,
                    pl.DataFrame(
                        {
                            "date": np.arange(len(values)),
                            "fold": ["train"] * len(train) + ["test"] * len(test),
                            "label": np.concatenate(
                                [np.zeros(len(train)), labels[:, 0]]
                            ).astype(np.int32),
                            **dict(zip(config.input_channels, values.T, strict=True)),
                        }
                    ),
                )
            )
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={},
            clients=clients,
            manifest_fields={
                "raw_files": [str(path) for path in cls.raw_files(config=config)],
                "machines": list(cls.MACHINES),
            },
        )
