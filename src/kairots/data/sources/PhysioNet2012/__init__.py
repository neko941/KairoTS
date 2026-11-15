"""PhysioNet/Computing in Cardiology Challenge 2012: ICU in-hospital mortality.

Sets A and B (4000 patients each) record the first 48 hours of an ICU stay as
`Time,Parameter,Value` rows: 36 measured variables, Weight, and the descriptors Age,
Gender and Height at time 0. Set A is the train fold and set B the test fold; the
label is `In-hospital_death`. One client per ICU type (`ICUType` 1 to 4). Time is in
hours since admission. Repeated measurements of a variable at one time are averaged;
-1 for Age, Gender, Height or Weight means not recorded.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Irregular import Irregular

URL_ROOT = "https://physionet.org/files/challenge-2012/1.0.0"
SETS = {"a": "train", "b": "test"}
ICU_TYPES = {1: "Coronary", 2: "CardiacSurgery", 3: "Medical", 4: "Surgical"}
UNRECORDED = ("Age", "Gender", "Height", "Weight")
VARIABLES = (
    *("ALP", "ALT", "AST", "Age", "Albumin", "BUN", "Bilirubin", "Cholesterol"),
    *("Creatinine", "DiasABP", "FiO2", "GCS", "Gender", "Glucose", "HCO3", "HCT"),
    *("HR", "Height", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP"),
    *("NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2", "SysABP"),
    *("Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight", "pH"),
)

# Ordered input used by CSDI's 48-hour probabilistic imputation experiment.
CSDI_VARIABLES = (
    "DiasABP",
    "HR",
    "Na",
    "Lactate",
    "NIDiasABP",
    "PaO2",
    "WBC",
    "pH",
    "Albumin",
    "ALT",
    "Glucose",
    "SaO2",
    "Temp",
    "AST",
    "Bilirubin",
    "HCO3",
    "BUN",
    "RespRate",
    "Mg",
    "HCT",
    "SysABP",
    "FiO2",
    "K",
    "GCS",
    "Cholesterol",
    "NISysABP",
    "TroponinT",
    "MAP",
    "TroponinI",
    "PaCO2",
    "Platelets",
    "Urine",
    "NIMAP",
    "Creatinine",
    "ALP",
)


@dataclass(kw_only=True)
class PhysioNet2012(Irregular):
    TASKS = ("Classification",)
    REQUIRES = {"data.scaler.imputation": ("masked",)}
    SHA256 = {
        "set-a.tar.gz": (
            "8cb250f179cd0952b4b9ebcf8954b63d70383131670fac1cfee13deaa13ca920"
        ),
        "set-b.tar.gz": (
            "b1637a2a423a8e76f8f087896cfc5fdf28f88519e1f4e874fbda69b2a64dac30"
        ),
        "Outcomes-a.txt": (
            "2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2"
        ),
        "Outcomes-b.txt": (
            "1305f29734da374a809483229a99f33a9230aa8d4fa74b1a2d6ad788136423d5"
        ),
    }

    @dataclass(kw_only=True)
    class Config(Irregular.Config):
        name: str = "PhysioNet2012"
        raw_path: str = "datasets/PhysioNet2012/raw"
        input_channels: tuple[str, ...] = ("time", *VARIABLES)
        # The most observation times of one patient in sets A and B.
        input_length: int = 203
        num_classes: int = 2

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        return [Path(config.raw_path) / name for name in cls.SHA256]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """The challenge files on PhysioNet."""
        return tuple(
            RemoteFile(path=name, urls=(f"{URL_ROOT}/{name}",), sha256=sha256)
            for name, sha256 in cls.SHA256.items()
        )

    @staticmethod
    def read_set(*, path: Path) -> pl.DataFrame:
        """All `Time,Parameter,Value` rows of a set archive, with the `record` id."""
        frames: list[pl.DataFrame] = []
        with tarfile.open(path) as bundle:
            for member in bundle.getmembers():
                handle = bundle.extractfile(member)
                if handle is None or not member.name.endswith(".txt"):
                    continue
                frame = pl.read_csv(
                    handle.read(), schema_overrides={"Value": pl.Float64}
                )
                frames.append(
                    frame.with_columns(record=pl.lit(int(Path(member.name).stem)))
                )
        return pl.concat(frames)

    @classmethod
    def load(cls, *, config: Config) -> PhysioNet2012:
        paths = dict(zip(cls.SHA256, cls.raw_files(config=config), strict=True))
        variables = [name for name in config.input_channels if name != "time"]
        observations: list[pl.DataFrame] = []
        outcomes: list[pl.DataFrame] = []
        for letter, fold in SETS.items():
            rows = cls.read_set(path=paths[f"set-{letter}.tar.gz"])
            observations.append(rows.with_columns(fold=pl.lit(fold)))
            outcomes.append(pl.read_csv(paths[f"Outcomes-{letter}.txt"]))
        rows = pl.concat(observations).filter(
            ~(pl.col("Parameter").is_in(UNRECORDED) & (pl.col("Value") == -1))
        )
        clock = pl.col("Time").str.split(":")
        rows = rows.with_columns(
            time=clock.list.get(0).cast(pl.Float64)
            + clock.list.get(1).cast(pl.Float64) / 60
        )
        icu = rows.filter(pl.col("Parameter") == "ICUType").select(
            "record", "fold", icu=pl.col("Value").cast(pl.Int64)
        )
        wide = (
            rows.filter(pl.col("Parameter").is_in(variables))
            .group_by("record", "fold", "time", "Parameter")
            .agg(pl.col("Value").mean())
            .pivot(on="Parameter", index=["record", "fold", "time"], values="Value")
        )
        absent = [name for name in variables if name not in wide.columns]
        wide = (
            wide.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(name) for name in absent
            )
            .sort("record", "time")
            .with_columns(step=pl.int_range(pl.len()).over("record"))
        )
        longest = wide.group_by("record").len().get_column("len").max()
        if not isinstance(longest, int) or longest > config.input_length:
            raise ValueError(
                f"a patient has {longest} observation times, config input_length "
                f"{config.input_length}"
            )
        labels = pl.concat(outcomes).select(
            record=pl.col("RecordID"), label=pl.col("In-hospital_death").cast(pl.Int32)
        )
        samples = (
            icu.join(labels, on="record")
            .sort(pl.col("fold") == "test", "record")
            .with_row_index("sample")
        )
        if samples.height != rows.get_column("record").n_unique():
            raise ValueError("every record needs an ICUType and an outcome")
        grid = (
            samples.join(
                pl.DataFrame({"step": range(config.input_length)}), how="cross"
            )
            .join(wide.drop("fold"), on=["record", "step"], how="left")
            .with_columns(pl.col("sample", "step").cast(pl.Int32))
        )
        clients = [
            (
                name,
                grid.filter(pl.col("icu") == code)
                .sort("sample", "step")
                .select("sample", "step", "fold", "label", *config.input_channels),
            )
            for code, name in ICU_TYPES.items()
        ]
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={},
            clients=clients,
            manifest_fields={
                "raw_files": [str(path) for path in paths.values()],
                "classes": ["survived", "died"],
            },
        )
