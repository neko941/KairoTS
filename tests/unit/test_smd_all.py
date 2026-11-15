import numpy as np
import polars as pl
from _pytest.monkeypatch import MonkeyPatch

from kairots.data.sources.SMD import SMD
from kairots.data.sources.SMD.SMDAll import SMDAll


def test_smd_all_pins_28_machines_and_84_source_files() -> None:
    source = SMDAll.Config(raw_path="raw")

    assert len(SMDAll.MACHINES) == 28
    assert len(SMDAll.raw_files(config=source)) == 84


def test_smd_all_concatenates_train_then_test_in_machine_order(
    monkeypatch: MonkeyPatch,
) -> None:
    columns = {"value": [1.0, 2.0, 3.0, 4.0]}
    first = pl.DataFrame(
        {
            "date": range(4),
            "fold": ["train", "train", "test", "test"],
            "label": [0, 0, 0, 1],
            **columns,
        }
    )
    second = pl.DataFrame(
        {
            "date": range(4),
            "fold": ["train", "train", "test", "test"],
            "label": [0, 0, 1, 0],
            "value": [5.0, 6.0, 7.0, 8.0],
        }
    )

    def load(cls: type[SMD], *, config: SMD.Config) -> SMD:
        return SMD(
            name="SMD",
            input_channels=config.input_channels,
            timestamps={},
            clients=[("machine-1", first), ("machine-2", second)],
            manifest_fields={},
        )

    monkeypatch.setattr(SMD, "load", classmethod(load))
    combined = SMDAll.load(config=SMDAll.Config(input_channels=("value",)))
    name, frame = combined.clients[0]

    assert name == "SMD"
    assert frame.get_column("value").to_list() == [1, 2, 5, 6, 3, 4, 7, 8]
    assert frame.get_column("fold").to_list() == [
        "train",
        "train",
        "train",
        "train",
        "test",
        "test",
        "test",
        "test",
    ]
    assert frame.get_column("label").to_list() == [0, 0, 0, 0, 0, 1, 1, 0]


def test_historical_validation_overlaps_train_but_keeps_test_disjoint() -> None:
    from kairots.data.splits.HistoricalValidation import HistoricalValidation
    from kairots.data.splits.Predefined import Predefined

    frame = pl.DataFrame(
        {"fold": ["train"] * 500 + ["test"] * 100, "label": np.zeros(600)}
    )
    clean = Predefined.Config(window_length=100, validation_ratio=0.2, train_stride=100)
    historical = HistoricalValidation.Config(
        window_length=100, validation_ratio=0.2, train_stride=100
    )
    clean_windows = Predefined(config=clean).series(
        frame=frame, input_length=0, output_length=100, offset_length=0
    )
    old_windows = HistoricalValidation(config=historical).series(
        frame=frame, input_length=0, output_length=100, offset_length=0
    )

    assert clean_windows["train"] == [(0, 0), (100, 100), (200, 200), (300, 300)]
    assert old_windows["train"] == clean_windows["train"] + [(400, 400)]
    assert old_windows["validation"] == [(400, 400)]
    assert old_windows["test"] == [(500, 500)]
