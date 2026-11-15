from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import torch

from kairots.config.schema import Experiment
from kairots.data.splits.Predefined import Predefined
from kairots.metrics.AdjustedF1 import AdjustedF1
from kairots.metrics.F1 import F1


def _tslib_adjustment(*, gt: list[int], pred: list[int]) -> list[int]:
    """TSLib utils/tools.adjustment, copied as the reference."""
    pred = list(pred)
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                elif pred[j] == 0:
                    pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                elif pred[j] == 0:
                    pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


def test_point_adjustment_matches_tslib() -> None:
    generator = np.random.default_rng(0)
    for _ in range(50):
        gt = (generator.random(200) < 0.3).astype(int).tolist()
        pred = (generator.random(200) < 0.1).astype(int).tolist()
        adjusted = AdjustedF1.adjust(
            prediction=torch.tensor(pred, dtype=torch.bool),
            target=torch.tensor(gt, dtype=torch.bool),
        )
        assert adjusted.int().tolist() == _tslib_adjustment(gt=gt, pred=pred)


def test_f1_merges_and_is_undefined_without_positives() -> None:
    first, second, pooled = F1(), F1(), F1()
    first.update(prediction=torch.tensor([1, 0, 1]), target=torch.tensor([1, 1, 0]))
    second.update(prediction=torch.tensor([0, 1]), target=torch.tensor([0, 1]))
    pooled.update(
        prediction=torch.tensor([1, 0, 1, 0, 1]), target=torch.tensor([1, 1, 0, 0, 1])
    )
    first.merge(other=second)
    assert first.compute() == pooled.compute() == 2 * 2 / (2 * 2 + 1 + 1)
    empty = F1()
    empty.update(prediction=torch.zeros(4), target=torch.zeros(4))
    assert math.isnan(empty.compute())


def test_series_windows_tile_and_never_cross_cut_or_fold() -> None:
    frame = pl.DataFrame({"fold": ["train"] * 100 + ["test"] * 55})
    splits = Predefined(
        config=Predefined.Config(validation_ratio=0.2, window_length=10)
    ).series(frame=frame, input_length=0, output_length=10, offset_length=0)
    assert [start for start, _ in splits["train"]] == list(range(0, 71, 10))
    assert [start for start, _ in splits["validation"]] == [80, 90]
    assert [start for start, _ in splits["test"]] == [100, 110, 120, 130, 140]
    assert all(start == target for start, target in sum(splits.values(), []))


def test_anomaly_source_rejects_other_tasks() -> None:
    with pytest.raises(ValueError, match="does not support task 'Forecasting'"):
        Experiment.from_dict(
            values={
                "task": {"name": "Forecasting"},
                "data": {"source": {"name": "SMDGroup1"}},
            }
        )
