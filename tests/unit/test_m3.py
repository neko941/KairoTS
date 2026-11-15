from dataclasses import dataclass
from pathlib import Path

import pytest

from kairots.data.sources.M3 import M3


class _Sheet:
    name = "Tiny"

    def __init__(self, rows: list[list[object]]) -> None:
        self.rows = rows
        self.nrows = len(rows)

    def cell_value(self, row: int, column: int) -> object:
        return self.rows[row][column]

    def row_values(self, row: int, start: int = 0) -> list[object]:
        return self.rows[row][start:]


class _Workbook:
    def __init__(self, sheet: _Sheet) -> None:
        self.sheet = sheet

    def sheet_by_name(self, name: str) -> _Sheet:
        assert name == self.sheet.name
        return self.sheet

    def release_resources(self) -> None:
        pass


@dataclass(kw_only=True)
class _TinyM3(M3):
    ABSTRACT = False
    SHEET = "Tiny"
    SERIES_COUNT = 2
    HORIZON = 2
    START_FIELDS = ("start_year",)


def _rows(*, bad_gap: bool = False) -> list[list[object]]:
    tail: list[object] = [1.0, 2.0, 3.0, 4.0]
    if bad_gap:
        tail[1] = ""
    return [
        ["id", "n", "nf", "category", "start", "unused", "v1", "v2", "v3", "v4"],
        ["Y1", 4, 2, "MICRO", 2000, "", *tail],
        ["Y2", 4, 2, "MICRO", 2001, "", 5.0, 6.0, 7.0, 8.0],
    ]


def test_m3_source_preserves_official_test_horizon_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "M3C.xls"
    path.touch()
    sheet = _Sheet(_rows())
    monkeypatch.setattr(
        "kairots.data.sources.M3.xlrd.open_workbook",
        lambda *_args, **_kwargs: _Workbook(sheet),
    )

    source = _TinyM3.load(config=_TinyM3.Config(raw_path=str(tmp_path)))

    assert source.clients[0][1].get_column("fold").to_list() == [
        "train",
        "train",
        "test",
        "test",
    ]
    assert source.clients[0][1].get_column("value").to_list() == [1, 2, 3, 4]
    assert source.manifest_fields["series"][0] == {
        "series_id": "Y1",
        "category": "MICRO",
        "start_year": 2000,
    }


def test_m3_source_rejects_gaps_in_declared_observations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "M3C.xls"
    path.touch()
    sheet = _Sheet(_rows(bad_gap=True))
    monkeypatch.setattr(
        "kairots.data.sources.M3.xlrd.open_workbook",
        lambda *_args, **_kwargs: _Workbook(sheet),
    )

    with pytest.raises(ValueError, match="contiguous observations"):
        _TinyM3.load(config=_TinyM3.Config(raw_path=str(tmp_path)))
