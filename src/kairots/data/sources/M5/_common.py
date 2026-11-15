"""Pinned raw files and parsing shared by the M5 source and WRMSSE metric."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIRECTORY_URL = (
    "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
)
FILE_IDS = {
    "calendar.csv": "1eeU8Me44yzgWsDD0dYrX4xVOixil_H4u",
    "sales_train_validation.csv": "1w5nrvtdZlIAsYdCHSmXoCq4rTzmhcuPL",
    "sales_test_validation.csv": "1qKCyL7ZK4DwzkuSC4xQi5NrA99wzpBfj",
    "weights_validation.csv": "1-yjyAXJeuUQWXloCi9yC1LNO7Qtl1hcH",
    "sales_train_evaluation.csv": "1-khP8tp1gRTfaQQV_PlKd6R_3Mw5KiBw",
    "sales_test_evaluation.csv": "1Bj7Xj15yn4j6-BM_mpSmOVTV0c2ihEdo",
    "weights_evaluation.csv": "1M37c__ioZChCYpY69oXSzBi3bKrKVEkN",
    "sell_prices.csv": "1YKo2D7rpg9y5L-qaidUjyJaKvTIR0rMI",
}
SHA256 = {
    "calendar.csv": "568d0fe5f41790142379698732908e4e57432c1c6396f3f59fb880a9c2b54231",
    "sales_train_validation.csv": (
        "a4bb02b2f900543fdc78f5b4b4dc32eb606ae78bbb98baf088a976aa7f8c1d07"
    ),
    "sales_test_validation.csv": (
        "964f8ff2c36ff8763fc1bfeb4753f081abf70923d0209e78bc4dd470d49e438a"
    ),
    "weights_validation.csv": (
        "c957ad6429e59ac148918ddca13656d71784957c38af19e3314cad5c931572ab"
    ),
    "sales_train_evaluation.csv": (
        "c21a519596680feb86f27a9e62f6c8b583f8be60c2c195f080ae8ca2990af2b7"
    ),
    "sales_test_evaluation.csv": (
        "73773250dd7f7464f7a1bf3a54874dc7fc8cbb49885cd0bf1efe09bf48fd930d"
    ),
    "weights_evaluation.csv": (
        "b430c288d92a0e55e08b76c20aafb4c538469c7ba95e47ab768e37e385299497"
    ),
    "sell_prices.csv": (
        "5c7cf045b1abbdc8d5007809c40b3682c5216075937f93e2cef01caddc4a30d9"
    ),
}
SERIES_COUNT = 30_490
LEVEL_COUNTS = (1, 3, 10, 3, 7, 9, 21, 30, 70, 3_049, 9_147, 30_490)
IDENTIFIERS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")


@dataclass(frozen=True)
class Sales:
    identifiers: tuple[tuple[str, ...], ...]
    days: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class Hierarchy:
    row_offsets: np.ndarray
    column_indices: np.ndarray
    weights: np.ndarray
    levels: np.ndarray
    bottom_count: int


def read_sales(
    *, path: Path, expected_days: int | None = None, series_count: int = SERIES_COUNT
) -> Sales:
    """Read one official M5 bottom-level sales CSV into a compact matrix."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        header = next(rows, None)
        if header is None or tuple(header[:5]) != IDENTIFIERS:
            raise ValueError(f"{path.name} has invalid M5 identifier columns")
        days = tuple(header[5:])
        if expected_days is not None and len(days) != expected_days:
            raise ValueError(
                f"{path.name} has {len(days)} day columns; expected {expected_days}"
            )
        values = np.empty((series_count, len(days)), dtype=np.float32)
        identifiers: list[tuple[str, ...]] = []
        for index, row in enumerate(rows):
            if index >= series_count or len(row) != len(header):
                raise ValueError(f"{path.name} has an invalid row at index {index}")
            identifiers.append(tuple(row[:5]))
            series = np.fromiter((float(value) for value in row[5:]), np.float32)
            if not np.isfinite(series).all() or (series < 0).any():
                raise ValueError(f"{path.name} has invalid unit sales at row {index}")
            values[index] = series
        if len(identifiers) != series_count or len(set(identifiers)) != series_count:
            raise ValueError(
                f"{path.name} has {len(identifiers)} unique series; "
                f"expected {series_count}"
            )
    return Sales(tuple(identifiers), days, values)


def aggregation(
    *,
    weights_path: Path,
    identifiers: tuple[tuple[str, ...], ...],
    level_counts: tuple[int, ...] = LEVEL_COUNTS,
) -> Hierarchy:
    """Build the official hierarchy as compressed row membership indices."""
    grouped: list[dict[tuple[str, ...], list[int]]] = [{} for _ in level_counts]
    for index, (item, department, category, store, state) in enumerate(identifiers):
        keys = (
            (),
            (state,),
            (store,),
            (category,),
            (department,),
            (state, category),
            (state, department),
            (store, category),
            (store, department),
            (item,),
            (state, item),
            (item, store),
        )
        for level, key in enumerate(keys):
            grouped[level].setdefault(key, []).append(index)

    row_indices: list[int] = []
    column_indices: list[int] = []
    weights: list[float] = []
    levels: list[int] = []
    with weights_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Level_id", "Agg_Level_1", "Agg_Level_2", "weight"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{weights_path.name} has an invalid M5 schema")
        counts = [0] * len(level_counts)
        level_weight = np.zeros(len(level_counts), dtype=np.float64)
        for row_index, row in enumerate(reader):
            level = int(row["Level_id"].removeprefix("Level"))
            if not 1 <= level <= 12:
                raise ValueError(f"invalid M5 aggregation level {level}")
            level_index = level - 1
            first, second = row["Agg_Level_1"], row["Agg_Level_2"]
            key = (
                (),
                (first,),
                (first,),
                (first,),
                (first,),
                (first, second),
                (first, second),
                (first, second),
                (first, second),
                (first,),
                (first, second),
                (first, second),
            )[level_index]
            members = grouped[level_index].get(key)
            if not members:
                raise ValueError(f"M5 weights row {row_index} has no sales series")
            weight = float(row["weight"])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"M5 weights row {row_index} has invalid weight")
            row_indices.extend([row_index] * len(members))
            column_indices.extend(members)
            weights.append(weight)
            levels.append(level)
            counts[level_index] += 1
            level_weight[level_index] += weight
    if tuple(counts) != level_counts:
        raise ValueError(
            f"M5 weights have level counts {counts}; expected {level_counts}"
        )
    if not np.allclose(level_weight, 1.0, atol=1e-5, rtol=0):
        raise ValueError(
            f"M5 weights must sum to one within each level: {level_weight}"
        )
    row_offsets = np.zeros(sum(level_counts) + 1, dtype=np.int64)
    np.add.at(row_offsets, np.asarray(row_indices, dtype=np.int64) + 1, 1)
    np.cumsum(row_offsets, out=row_offsets)
    return Hierarchy(
        row_offsets=row_offsets,
        column_indices=np.asarray(column_indices, dtype=np.int32),
        weights=np.asarray(weights, dtype=np.float64),
        levels=np.asarray(levels, dtype=np.int8),
        bottom_count=len(identifiers),
    )


def aggregate(*, hierarchy: Hierarchy, values: np.ndarray) -> np.ndarray:
    """Sum bottom-level `[time, series]` values into the 42,840 official rows."""
    if values.ndim != 2 or values.shape[1] != hierarchy.bottom_count:
        raise ValueError("M5 values do not match the hierarchy's bottom channels")
    result = np.empty((values.shape[0], len(hierarchy.weights)), dtype=np.float32)
    for row, (start, end) in enumerate(
        zip(hierarchy.row_offsets[:-1], hierarchy.row_offsets[1:], strict=True)
    ):
        result[:, row] = values[:, hierarchy.column_indices[start:end]].sum(axis=1)
    return result


def rmsse_scales(*, hierarchy: Hierarchy, history: np.ndarray) -> np.ndarray:
    """Official M5 RMSSE denominator, starting at each series' first positive day."""
    if history.ndim != 2 or history.shape[1] != hierarchy.bottom_count:
        raise ValueError("M5 history must have one row per bottom-level series")
    scales = np.empty(len(hierarchy.weights), dtype=np.float64)
    for row, (start, end) in enumerate(
        zip(hierarchy.row_offsets[:-1], hierarchy.row_offsets[1:], strict=True)
    ):
        series = history[:, hierarchy.column_indices[start:end]].sum(axis=1)
        observed = np.flatnonzero(series > 0)
        if not len(observed):
            raise ValueError(f"M5 hierarchy series {row} has no sales")
        changes = np.diff(series[int(observed[0]) :]).astype(np.float64)
        if not len(changes):
            raise ValueError(f"M5 hierarchy series {row} has no scale")
        scale = float(np.mean(changes**2))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"M5 hierarchy series {row} has zero RMSSE scale")
        scales[row] = scale
    return scales
