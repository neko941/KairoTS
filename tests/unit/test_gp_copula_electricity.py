from datetime import datetime

import pytest

from kairots.data.sources.GPCopula.ElectricityNIPS import _build_frame


def test_gpcopula_grouping_matches_mean_and_zero_padding_and_rolling_dates() -> None:
    train = [
        {
            "feat_static_cat": [0],
            "start": "2014-01-01 00:00:00",
            "target": [1, 2, 3, 4],
        },
        {
            "feat_static_cat": [1],
            "start": "2014-01-01 01:00:00",
            "target": [10, 20, 30],
        },
    ]
    test = [
        {
            "feat_static_cat": [0],
            "start": "2014-01-01 00:00:00",
            "target": [1, 2, 3, 4, 5, 6],
        },
        {
            "feat_static_cat": [1],
            "start": "2014-01-01 01:00:00",
            "target": [10, 20, 30, 40, 50],
        },
        {
            "feat_static_cat": [0],
            "start": "2014-01-01 00:00:00",
            "target": [1, 2, 3, 4, 5, 6, 7, 8],
        },
        {
            "feat_static_cat": [1],
            "start": "2014-01-01 01:00:00",
            "target": [10, 20, 30, 40, 50, 60, 70],
        },
    ]

    frame, timestamps, train_length = _build_frame(
        train_entries=train,
        test_entries=test,
        channels=("a", "b"),
        output_length=2,
        rolling_dates=2,
    )

    assert train_length == 4
    assert timestamps[0] == datetime(2014, 1, 1)
    assert frame.get_column("b").to_list() == [20, 10, 20, 30, 40, 50, 60, 70]
    assert frame.get_column("fold").to_list() == ["train"] * 4 + ["test"] * 4


def test_gpcopula_grouping_rejects_nonmatching_train_test_overlap() -> None:
    train = [
        {"feat_static_cat": [0], "start": "2014-01-01 00:00:00", "target": [1, 2]},
    ]
    test = [
        {"feat_static_cat": [0], "start": "2014-01-01 00:00:00", "target": [1, 3, 4]},
    ]

    with pytest.raises(ValueError, match="train/test overlap"):
        _build_frame(
            train_entries=train,
            test_entries=test,
            channels=("a",),
            output_length=1,
            rolling_dates=1,
        )
