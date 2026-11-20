import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from kairots.data.sources.M5 import M5
from kairots.data.sources.M5._common import aggregate, aggregation, rmsse_scales
from kairots.metrics.M5WRMSSE import M5WRMSSE, M5Context


@dataclass(kw_only=True)
class _TinyM5(M5):
    ABSTRACT = False
    FOLD = "validation"
    TRAIN_DAYS = 3
    SERIES_COUNT = 2
    FORECAST_LENGTH = 2

    @dataclass(kw_only=True)
    class Config(M5.Config):
        name: str = "_TinyM5"
        input_channels: tuple[str, ...] = ("series_a", "series_b")


def _write_sales(path: Path, *, days: tuple[int, ...], bad_id: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
                *[f"d_{day}" for day in days],
            ]
        )
        writer.writerow(
            [
                "i1",
                "d1",
                "c1",
                "s1",
                "st1",
                *([0, 1, 2] if days == (1, 2, 3) else [3, 4]),
            ]
        )
        writer.writerow(
            [
                "i2",
                "d2",
                "c2",
                "s2",
                "st2" if not bad_id else "st3",
                *([0, 0, 1] if days == (1, 2, 3) else [2, 3]),
            ]
        )


def test_m5_source_loads_contiguous_bottom_series_and_holdout(tmp_path: Path) -> None:
    _write_sales(tmp_path / "sales_train_validation.csv", days=(1, 2, 3))
    _write_sales(tmp_path / "sales_test_validation.csv", days=(4, 5))

    source = _TinyM5.load(config=_TinyM5.Config(raw_path=str(tmp_path)))

    assert source.input_channels == ("series_a", "series_b")
    assert len(source.clients) == 1
    _, frame = source.clients[0]
    assert frame.get_column("fold").to_list() == ["train"] * 3 + ["test"] * 2
    assert frame.get_column("series_a").to_list() == [0, 1, 2, 3, 4]
    assert source.manifest_fields["forecast_horizon"] == 2
    assert source.manifest_fields["series_metadata"]["series_b"]["item_id"] == "i2"


def test_m5_source_rejects_row_mismatch_and_noncontiguous_test(tmp_path: Path) -> None:
    _write_sales(tmp_path / "sales_train_validation.csv", days=(1, 2, 3))
    _write_sales(tmp_path / "sales_test_validation.csv", days=(4, 5), bad_id=True)
    with pytest.raises(ValueError, match="different series IDs"):
        _TinyM5.load(config=_TinyM5.Config(raw_path=str(tmp_path)))

    _write_sales(tmp_path / "sales_test_validation.csv", days=(5, 6))
    with pytest.raises(ValueError, match="not contiguous"):
        _TinyM5.load(config=_TinyM5.Config(raw_path=str(tmp_path)))


def _write_weights(path: Path) -> None:
    rows: list[list[str | float]] = [
        ["Level_id", "Agg_Level_1", "Agg_Level_2", "Dollar_Sales", "weight"]
    ]
    identifiers = (
        ("i1", "d1", "c1", "s1", "st1"),
        ("i2", "d2", "c2", "s2", "st2"),
    )
    keys = (
        lambda x: (),
        lambda x: (x[4],),
        lambda x: (x[3],),
        lambda x: (x[2],),
        lambda x: (x[1],),
        lambda x: (x[4], x[2]),
        lambda x: (x[4], x[1]),
        lambda x: (x[3], x[2]),
        lambda x: (x[3], x[1]),
        lambda x: (x[0],),
        lambda x: (x[4], x[0]),
        lambda x: (x[0], x[3]),
    )
    for level, key_for in enumerate(keys, start=1):
        groups: dict[tuple[str, ...], None] = {}
        for identifier in identifiers:
            groups[key_for(identifier)] = None
        for key in groups:
            first = key[0] if key else ""
            second = key[1] if len(key) > 1 else ""
            rows.append(
                [f"Level{level}", first, second, 1.0, 1.0 if level == 1 else 0.5]
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_m5_wrmsse_matches_manual_12_level_weighted_rmsse(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights_validation.csv"
    _write_weights(weights_path)
    identifiers = (
        ("i1", "d1", "c1", "s1", "st1"),
        ("i2", "d2", "c2", "s2", "st2"),
    )
    hierarchy = aggregation(
        weights_path=weights_path,
        identifiers=identifiers,
        level_counts=(1, *(2 for _ in range(11))),
    )
    history = np.asarray([[0, 0], [1, 0], [2, 1], [4, 3]], dtype=np.float32)
    scales = rmsse_scales(hierarchy=hierarchy, history=history)
    context = M5Context(
        row_offsets=hierarchy.row_offsets,
        column_indices=hierarchy.column_indices,
        weights=hierarchy.weights,
        levels=hierarchy.levels,
        bottom_count=2,
        scales=scales,
        forecast_length=2,
    )
    prediction = torch.tensor([[[5.0, 2.0], [6.0, 4.0]]])
    target = torch.tensor([[[4.0, 3.0], [4.0, 4.0]]])
    forecast_aggregate = aggregate(hierarchy=hierarchy, values=prediction[0].numpy())
    target_aggregate = aggregate(hierarchy=hierarchy, values=target[0].numpy())
    rmsse = np.sqrt(
        np.mean((forecast_aggregate - target_aggregate) ** 2, axis=0) / scales
    )
    expected = np.mean(
        [
            np.dot(
                hierarchy.weights[hierarchy.levels == level],
                rmsse[hierarchy.levels == level],
            )
            for level in range(1, 13)
        ]
    )

    metric = M5WRMSSE(config=M5WRMSSE.Config())
    metric.update(prediction=prediction, target=target, wrmsse_context=context)

    assert metric.compute() == pytest.approx(expected)


def test_wrmsse_family_scores_any_hierarchy() -> None:
    from kairots.metrics.WRMSSE import WRMSSE, HierarchyContext
    from kairots.registry import names

    # Two bottom series and their total: level 1 = total, level 2 = bottoms.
    context = HierarchyContext(
        row_offsets=np.asarray([0, 2, 3, 4]),
        column_indices=np.asarray([0, 1, 0, 1]),
        weights=np.asarray([1.0, 0.25, 0.75]),
        levels=np.asarray([1, 2, 2]),
        bottom_count=2,
        scales=np.asarray([4.0, 1.0, 1.0]),
        forecast_length=2,
    )
    prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.tensor([[[1.0, 1.0], [1.0, 5.0]]])
    # Errors: total (1, 1) -> mse 1 / scale 4; bottom 0 (0, 2) -> 2; bottom 1
    # (1, -1) -> 1.
    total = np.sqrt(1.0 / 4.0)
    bottoms = 0.25 * np.sqrt(2.0) + 0.75 * 1.0
    metric = WRMSSE(config=WRMSSE.Config())
    metric.update(prediction=prediction, target=target, wrmsse_context=context)
    assert metric.compute() == pytest.approx((total + bottoms) / 2)
    assert issubclass(M5WRMSSE, WRMSSE)
    assert "WRMSSE" not in names(category="metrics")
    assert "M5WRMSSE" in names(category="metrics")
    with pytest.raises(NotImplementedError):
        WRMSSE.prepare_context(experiment=None)
