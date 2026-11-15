from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from kairots import Experiment, run
from kairots.data.calendar import features
from kairots.models.TiDE import TiDE
from kairots.tasks.Forecasting import Forecasting

CHANNELS = ("HUFL", "HULL", "OT")


def _task(*, calendar: tuple[str, ...] = ("HourOfDay",)) -> Forecasting:
    return Forecasting(
        config=Forecasting.Config(
            target_channels=("OT",), calendar=calendar, input_channels=CHANNELS
        )
    )


def test_calendar_features_match_tslib_time_features() -> None:
    # 2016-07-04 13:00 is a Monday, day 186 of a leap year.
    dates = pl.Series([datetime(2016, 7, 4, 13)])
    names = ("HourOfDay", "DayOfWeek", "DayOfMonth", "DayOfYear", "MonthOfYear")
    expected = [13 / 23 - 0.5, -0.5, 3 / 30 - 0.5, 185 / 365 - 0.5, 6 / 11 - 0.5]
    assert features(dates=dates, names=names)[0].tolist() == pytest.approx(expected)
    with pytest.raises(ValueError, match="datetime"):
        features(dates=pl.Series([0, 1]), names=("HourOfDay",))


def test_timegrad_business_day_fourier_features_match_pinned_author_formula() -> None:
    dates = pl.Series([datetime(2020, 1, 1), datetime(2020, 12, 31)])
    result = features(dates=dates, names=("DayOfYearCos", "DayOfYearSin"))
    angle = 366 * 2 * math.pi / 367
    assert result[-1].tolist() == pytest.approx([math.cos(angle), math.sin(angle)])


def test_day_of_year_fourier_rejects_empty_dates() -> None:
    with pytest.raises(ValueError, match="at least one date"):
        features(dates=pl.Series("date", [], dtype=pl.Date), names=("DayOfYearCos",))


def test_future_values_of_targets_and_past_covariates_never_reach_the_model() -> None:
    task = _task()
    # Values columns: HUFL, HULL, OT, then one calendar column.
    batch_x = torch.randn(2, 5, 4)
    batch_y = torch.randn(2, 3, 4)
    poisoned = batch_y.clone()
    poisoned[..., :3] = 1e6
    inputs, target = task.prepare(batch_x=batch_x, batch_y=batch_y, output_length=3)
    leaked, _ = task.prepare(batch_x=batch_x, batch_y=poisoned, output_length=3)
    # [targets | past | calendar] over history and horizon.
    assert inputs.shape == (2, 8, 4)
    assert torch.equal(inputs[:, :5], batch_x[..., [2, 0, 1, 3]])
    assert inputs[:, 5:, :3].isnan().all()
    assert torch.equal(inputs[:, 5:, 3], batch_y[..., 3])
    assert torch.equal(inputs.nan_to_num(), leaked.nan_to_num())
    assert torch.equal(target, batch_y[..., [2]])


def test_target_only_task_keeps_every_channel() -> None:
    task = Forecasting(config=Forecasting.Config(input_channels=CHANNELS))
    batch_x, batch_y = torch.randn(2, 5, 3), torch.randn(2, 4, 3)
    inputs, target = task.prepare(batch_x=batch_x, batch_y=batch_y, output_length=3)
    assert inputs is batch_x and torch.equal(target, batch_y[:, -3:])


def test_tide_reads_no_unavailable_value() -> None:
    torch.manual_seed(0)
    model = TiDE(
        config=TiDE.Config(
            input_length=5,
            output_length=3,
            input_channels=3,
            output_channels=1,
            calendar_channels=1,
            hidden=8,
            temporal_hidden=8,
        )
    ).eval()
    inputs, _ = _task().prepare(
        batch_x=torch.randn(2, 5, 4), batch_y=torch.randn(2, 3, 4), output_length=3
    )
    prediction = model(batch_x=inputs)
    assert prediction.shape == (2, 3, 1) and prediction.isfinite().all()
    assert torch.equal(prediction, model(batch_x=inputs.nan_to_num(nan=1e6)))


def _experiment(*, root: Path, model: str) -> dict:
    return {
        "task": {
            "name": "Forecasting",
            "target_channels": ["OT"],
            "calendar": ["HourOfDay", "DayOfWeek"],
        },
        "data": {
            "source": {
                "name": "ETDatasetHour",
                "raw_path": str(root / "raw"),
                "download": False,
                "input_channels": list(CHANNELS),
                "batch_size": 16,
            },
            "split": {"input_length": 8, "output_length": 3, "min_windows": 2},
            "store": {"root": str(root / "datasets")},
        },
        "model": {"name": model},
        "strategy": {"iterations": 1},
        "evaluation": {"metrics": {"test": ["MAE", "MASE"]}},
        "runtime": {
            "device": "cpu",
            "runs_dir": str(root / "runs"),
            "save_checkpoints": False,
        },
    }


def test_covariates_need_a_model_that_declares_them(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="takes no covariates"):
        Experiment.from_dict(values=_experiment(root=tmp_path, model="Linear"))
    experiment = Experiment.from_dict(values=_experiment(root=tmp_path, model="TiDE"))
    assert experiment.task.input_channels == CHANNELS
    assert (experiment.model.output_channels, experiment.model.calendar_channels) == (
        1,
        2,
    )


def test_tide_run_scores_the_target_channel(workspace_tmp_path: Path) -> None:
    start = datetime(2016, 7, 1)
    for variant, offset in (("ETTh1", 0.0), ("ETTh2", 2.0)):
        directory = workspace_tmp_path / "raw" / variant
        directory.mkdir(parents=True)
        time = np.arange(240)
        pl.DataFrame(
            {
                "date": [start + timedelta(hours=int(step)) for step in time],
                **{
                    channel: (np.sin(time / 9 + offset + index) * 10 + 50).tolist()
                    for index, channel in enumerate(
                        ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
                    )
                },
            }
        ).write_csv(directory / f"{variant}.csv")
    result = run(
        experiment=Experiment.from_dict(
            values=_experiment(root=workspace_tmp_path, model="TiDE")
        )
    )
    last = result.rounds[-1]
    assert math.isfinite(last["generalization_test_pooled_mse"]) and math.isfinite(
        last["generalization_test_pooled_mase"]
    )
