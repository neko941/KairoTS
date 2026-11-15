import csv
from dataclasses import dataclass
from pathlib import Path

import pytest

from kairots.data.sources.M4 import M4


@dataclass(kw_only=True)
class _TinyM4(M4):
    ABSTRACT = False
    PERIOD = "Tiny"
    SERIES_COUNT = 2
    HORIZON = 2
    SEASONALITY = 1


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)


def _write_m4(
    *,
    root: Path,
    broken_id: bool = False,
    train_gap: bool = False,
    duplicate_metadata: bool = False,
) -> None:
    _write_csv(
        root / "Train/Tiny-train.csv",
        [["M4id", "V1", "V2", "V3"], ["T1", "1", "", "3"], ["T2", "4", "5", ""]]
        if train_gap
        else [["M4id", "V1", "V2"], ["T1", "1", "2"], ["T2", "4", "5"]],
    )
    _write_csv(
        root / "Test/Tiny-test.csv",
        [
            ["M4id", "F1", "F2"],
            ["T1", "6", "7"],
            ["T3" if broken_id else "T2", "8", "9"],
        ],
    )
    info_rows = [
        ["M4id", "category", "Frequency", "Horizon", "SP", "StartingDate"],
        ["T1", "Macro", "1", "2", "Tiny", "01-01-00"],
        ["T2", "Finance", "1", "2", "Tiny", "01-01-01"],
    ]
    if duplicate_metadata:
        info_rows.append(info_rows[1])
    _write_csv(root / "M4-info.csv", info_rows)


def test_m4_source_preserves_official_folds_and_series_metadata(
    tmp_path: Path,
) -> None:
    _write_m4(root=tmp_path)

    source = _TinyM4.load(config=_TinyM4.Config(raw_path=str(tmp_path)))

    first = source.clients[0][1]
    assert first.get_column("fold").to_list() == ["train", "train", "test", "test"]
    assert first.get_column("value").to_list() == [1.0, 2.0, 6.0, 7.0]
    assert source.manifest_fields["series_metadata"]["T1"] == {
        "category": "Macro",
        "frequency": 1,
        "starting_date": "01-01-00",
    }


@pytest.mark.parametrize(
    "broken_id,train_gap,duplicate_metadata",
    [(True, False, False), (False, True, False), (False, False, True)],
)
def test_m4_source_rejects_mismatched_ids_and_interior_missing_values(
    tmp_path: Path,
    *,
    broken_id: bool,
    train_gap: bool,
    duplicate_metadata: bool,
) -> None:
    _write_m4(
        root=tmp_path,
        broken_id=broken_id,
        train_gap=train_gap,
        duplicate_metadata=duplicate_metadata,
    )

    with pytest.raises(ValueError):
        _TinyM4.load(config=_TinyM4.Config(raw_path=str(tmp_path)))
