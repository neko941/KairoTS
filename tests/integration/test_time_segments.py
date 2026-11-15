"""TimeSegments: consecutive time tasks; leakage, P matrix and replay checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from fixtures.synthetic import ett_experiment

from kairots.config.schema import Experiment
from kairots.data.stores.Parquet import Parquet
from kairots.metrics._continual import forecasting_summary
from kairots.runtime.orchestrator import run

LENGTH = 80
PERIODS = 3
CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def _write(*, path: Path, poison: tuple[int, int] | None = None) -> None:
    """Two ETT series whose frequency changes with every task; optionally poisoned."""
    for variant, offset in (("ETTh1", 0.0), ("ETTh2", 2.0)):
        (path / variant).mkdir(parents=True)
        time = np.arange(LENGTH * PERIODS)
        frequency = np.array([1 / 9, 1 / 4, 1 / 2.5])[time // LENGTH]
        columns = {
            channel: np.sin(time * frequency + offset + index)
            for index, channel in enumerate(CHANNELS)
        }
        if poison is not None:
            for values in columns.values():
                values[poison[0] : poison[1]] += 1000.0
        pl.DataFrame({"date": time.tolist(), **columns}).write_csv(
            path / variant / f"{variant}.csv"
        )


def _values(
    *, root: Path, raw: Path, replay_size: int = 0, iterations: int = 2
) -> dict[str, Any]:
    values = ett_experiment(
        raw_path=raw,
        runs_dir=root / "runs",
        iterations=iterations,
        save_checkpoints=True,
    ).to_dict()
    values["data"]["split"] = {
        "name": "TaskWindowRatio",
        "input_length": 8,
        "output_length": 3,
        "ratios": [0.8, 0.0, 0.2],
        "min_windows": 2,
        "base_length": LENGTH,
        "task_length": LENGTH,
    }
    values["data"]["store"]["root"] = str(root / "datasets")
    values["model"] = {"name": "DLinear"}
    values["training"] = {
        "name": "TimeSegments",
        "periods": PERIODS,
        "base_length": LENGTH,
        "task_length": LENGTH,
        "replay_size": replay_size,
        "capacity": 128,
    }
    return values


def _run(*, values: dict[str, Any]) -> Any:
    return run(experiment=Experiment.from_dict(values=values))


def _train_losses(*, result: Any) -> list[tuple[int, int, float]]:
    lines = (Path(result.run_dir) / "clients.jsonl").read_text().splitlines()
    return [
        (row["current_iter"], row["client_id"], row["train_loss"])
        for row in map(json.loads, lines)
    ]


def test_forecasting_summary_matches_hand_computation() -> None:
    matrix = [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0]]
    summary = forecasting_summary(matrix=matrix)
    assert summary["AF"] == pytest.approx(((4 - 1) + (5 - 3)) / 2)
    assert summary["AP"] == pytest.approx((1 + 3 + 6) / 3)
    assert summary["AvgPerf"] == pytest.approx((4 + 5 + 6) / 3)


def test_earlier_tasks_never_see_future_task_rows(workspace_tmp_path: Path) -> None:
    clean_raw, poisoned_raw = workspace_tmp_path / "a", workspace_tmp_path / "b"
    _write(path=clean_raw)
    _write(path=poisoned_raw, poison=(LENGTH, LENGTH * PERIODS))
    clean = _run(values=_values(root=workspace_tmp_path / "c", raw=clean_raw))
    poisoned = _run(values=_values(root=workspace_tmp_path / "d", raw=poisoned_raw))
    task_zero = [row for row in clean.rounds if row["period"] == 0]
    assert task_zero
    for before, after in zip(task_zero, poisoned.rounds, strict=False):
        assert before["period_0_pooled_mse"] == after["period_0_pooled_mse"]
    first = [item for item in _train_losses(result=clean) if item[0] < 2]
    assert first == [item for item in _train_losses(result=poisoned) if item[0] < 2]
    # The poison reaches the later tasks, so it is a real perturbation.
    assert (
        clean.rounds[-1]["period_1_pooled_mse"]
        != (poisoned.rounds[-1]["period_1_pooled_mse"])
    )


@pytest.mark.parametrize("period", range(PERIODS))
def test_training_and_scaler_ignore_every_task_test_segment(
    workspace_tmp_path: Path,
    period: int,
) -> None:
    clean_raw, poisoned_raw = workspace_tmp_path / "a", workspace_tmp_path / "b"
    _write(path=clean_raw)
    # The last 20% of every task is its test segment.
    end = (period + 1) * LENGTH
    _write(path=poisoned_raw, poison=(end - 12, end))
    clean = _run(values=_values(root=workspace_tmp_path / "c", raw=clean_raw))
    poisoned = _run(values=_values(root=workspace_tmp_path / "d", raw=poisoned_raw))
    assert _train_losses(result=clean) == _train_losses(result=poisoned)
    assert (
        clean.rounds[-1][f"period_{period}_pooled_mse"]
        != poisoned.rounds[-1][f"period_{period}_pooled_mse"]
    )


def test_p_matrix_is_lower_triangular_and_ordered_by_task(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    _write(path=raw)
    result = _run(values=_values(root=workspace_tmp_path, raw=raw))
    summary = json.loads((Path(result.run_dir) / "continual.json").read_text())
    matrix = summary["P"]
    assert [len(row) for row in matrix] == [1, 2, 3]
    ends = [result.rounds[1], result.rounds[3], result.rounds[5]]
    for task, (row, end) in enumerate(zip(matrix, ends, strict=True)):
        assert end["period"] == task
        for evaluated, value in enumerate(row):
            assert value == pytest.approx(end[f"period_{evaluated}_pooled_mse"] ** 0.5)
    for name, value in forecasting_summary(matrix=matrix).items():
        assert summary[name] == pytest.approx(value)


def test_replay_forgets_less_than_naive(workspace_tmp_path: Path) -> None:
    raw = workspace_tmp_path / "raw"
    _write(path=raw)
    forgetting = {}
    for replay_size in (0, 128):
        values = _values(
            root=workspace_tmp_path / str(replay_size),
            raw=raw,
            replay_size=replay_size,
            iterations=30,
        )
        values["learner"]["learning_rate"] = 0.01
        result = _run(values=values)
        summary = json.loads((Path(result.run_dir) / "continual.json").read_text())
        forgetting[replay_size] = summary["AF"]
    assert forgetting[128] < forgetting[0]


def test_explicit_ranges_exclude_unused_rows_from_training_and_scaling(
    workspace_tmp_path: Path,
) -> None:
    task_ranges = [[10, 70], [90, 150], [170, 230]]
    results = []
    for label, poison in (("clean", None), ("poisoned", (0, 10))):
        raw = workspace_tmp_path / label / "raw"
        _write(path=raw, poison=poison)
        values = _values(root=workspace_tmp_path / label, raw=raw)
        values["data"]["split"]["task_ranges"] = task_ranges
        values["training"]["task_ranges"] = task_ranges
        experiment = Experiment.from_dict(values=values)
        _, records = Parquet(config=experiment.data.store).prepare(
            experiment=experiment
        )
        for record in records:
            _, splits = Parquet.load(record=record)
            assert min(start for start, _ in splits["train"]) == 10
            for windows in splits.values():
                assert all(
                    any(
                        first <= start and target + 3 <= last
                        for first, last in task_ranges
                    )
                    for start, target in windows
                )
        results.append(_run(values=values))
    assert _train_losses(result=results[0]) == _train_losses(result=results[1])
    summaries = [
        json.loads((Path(result.run_dir) / "continual.json").read_text())
        for result in results
    ]
    assert summaries[0] == summaries[1]


@pytest.mark.parametrize(
    "task_ranges",
    [((5, 3),), ((0, 20), (19, 30)), ((-1, 10),), ((0, 10.5),), ((0,),), (10,)],
)
def test_invalid_task_ranges_are_rejected(
    workspace_tmp_path: Path, task_ranges: Any
) -> None:
    values = _values(root=workspace_tmp_path, raw=workspace_tmp_path / "raw")
    values["data"]["split"]["task_ranges"] = task_ranges
    values["training"]["task_ranges"] = task_ranges
    with pytest.raises(ValueError, match="task_ranges"):
        Experiment.from_dict(values=values)


def test_task_range_count_and_schedule_must_agree(workspace_tmp_path: Path) -> None:
    values = _values(root=workspace_tmp_path, raw=workspace_tmp_path / "raw")
    values["data"]["split"]["task_ranges"] = [[0, 80], [80, 160]]
    values["training"]["task_ranges"] = [[0, 80], [80, 160]]
    with pytest.raises(ValueError, match="task_ranges"):
        Experiment.from_dict(values=values)
    values["training"]["task_ranges"] = [[0, 80], [80, 160], [160, 240]]
    with pytest.raises(ValueError, match="task_ranges"):
        Experiment.from_dict(values=values)


def test_explicit_ranges_must_fit_the_data(workspace_tmp_path: Path) -> None:
    raw = workspace_tmp_path / "raw"
    _write(path=raw)
    values = _values(root=workspace_tmp_path, raw=raw)
    values["data"]["split"]["task_ranges"] = [[0, 80], [80, 160], [160, 250]]
    values["training"]["task_ranges"] = [[0, 80], [80, 160], [160, 250]]
    with pytest.raises(RuntimeError, match="task_ranges exceed the available rows"):
        _run(values=values)
