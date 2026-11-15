"""EthanolConcentration: spectra of water-ethanol solutions in whisky bottles.

261 train and 263 test samples, 3 repeat readings of one bottle, length 1751
(226 nm to 1101.5 nm every 0.5 nm), 4 ethanol concentrations. The archive is the
official UEA multivariate release, mirrored on Zenodo (record 11206212).
"""

from dataclasses import dataclass

from kairots.data.download import RemoteFile
from kairots.data.sources.UEA import FOLDS, UEA

ZENODO = "https://zenodo.org/api/records/11206212/files"


class EthanolConcentration(UEA):
    TASKS = ("Classification",)
    PROBLEM = "EthanolConcentration"
    SHA256 = {
        "TRAIN": "3255f7f6c2e0f3f6b4dbb53fee55460baf60c634de54088d6f69660f0b360ba8",
        "TEST": "66a1ed323a73366b7544b3f980c1a33722beaaabecb87a6a060755ea08a96913",
    }

    @dataclass(kw_only=True)
    class Config(UEA.Config):
        name: str = "EthanolConcentration"
        input_channels: tuple[str, ...] = ("reading_1", "reading_2", "reading_3")
        input_length: int = 1751
        num_classes: int = 4

    @classmethod
    def remote_files(cls, *, config: UEA.Config) -> tuple[RemoteFile, ...]:
        return tuple(
            RemoteFile(
                path=f"{cls.PROBLEM}/{cls.PROBLEM}_{fold}.ts",
                urls=(f"{ZENODO}/{cls.PROBLEM}_{fold}.ts/content",),
                sha256=cls.SHA256[fold],
            )
            for fold in FOLDS
        )
