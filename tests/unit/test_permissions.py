from pathlib import Path

import polars as pl
import pytest

from kairots.data.contracts import ClientRecord
from kairots.data.stores.Parquet import Parquet
from kairots.runtime.permissions import DataPermissions


def test_data_roles_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="only one permission role"):
        DataPermissions(source=("train",), query=("train",))


def test_fit_phases_are_source_and_support_only() -> None:
    permissions = DataPermissions(source=("train",), support=("validation",))
    assert permissions.fit_phases == ("train", "validation")
    assert permissions.query == ()


def test_fit_panel_load_hides_query_samples_and_labels(tmp_path: Path) -> None:
    path = tmp_path / "panel.parquet"
    pl.DataFrame(
        {
            "sample": [0, 0, 1, 1, 2, 2],
            "label": [0, 0, 1, 1, 2, 2],
            "x": [0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
        }
    ).write_parquet(path)
    record = ClientRecord(
        client_id=0,
        name="panel",
        store="Parquet",
        values_path=str(path),
        windows={"train": ((0, 0),), "validation": ((1, 1),), "test": ((2, 2),)},
        target_offset=0,
        channels=("x",),
        mean=(0.0,),
        scale=(1.0,),
        permissions=("train", "validation"),
    )

    values, splits = Parquet.load(record=record)

    assert values.shape == (2, 2, 1)
    assert values[:, :, 0].tolist() == [[0.0, 1.0], [10.0, 11.0]]
    assert splits["train"] == [(0, 0)]
    assert splits["validation"] == [(1, 1)]
    assert splits["test"] == []
