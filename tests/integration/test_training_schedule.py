"""Offline and class-incremental schedules share the same fitting path."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from integration.test_classification import _config, _raw
from integration.test_continual import configuration
from kairots.config.schema import Experiment
from kairots.registry import get
from kairots.runtime.orchestrator import run


def _parameters(*, directory: str) -> dict[str, Any]:
    return torch.load(Path(directory) / "checkpoints/final.pt", weights_only=True)


@pytest.mark.parametrize(
    "task,strategy",
    [
        ("Forecasting", "FedAvg"),
        ("Forecasting", "Centralized"),
        ("Classification", "FedAvg"),
    ],
)
def test_offline_schedule_matches_implicit_full_training(
    workspace_tmp_path: Path, task: str, strategy: str
) -> None:
    if task == "Forecasting":
        raw = workspace_tmp_path / "raw"
        write_ett_sources(path=raw)
        values = ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / "runs",
            strategy=strategy,
            iterations=2,
            save_checkpoints=True,
        ).to_dict()
    else:
        raw = _raw(root=workspace_tmp_path / "raw")
        values = _config(root=workspace_tmp_path, raw=raw, iterations=2)
        values["runtime"]["save_checkpoints"] = True
    implicit = run(experiment=Experiment.from_dict(values=values))
    values["training"] = {"name": "Offline"}
    explicit = run(experiment=Experiment.from_dict(values=values))
    first = _parameters(directory=implicit.run_dir)
    last = _parameters(directory=explicit.run_dir)
    assert first.keys() == last.keys()
    assert all(torch.equal(value, last[key]) for key, value in first.items())
    assert [
        {key: value for key, value in row.items() if not key.endswith("_seconds")}
        for row in implicit.rounds
    ] == [
        {key: value for key, value in row.items() if not key.endswith("_seconds")}
        for row in explicit.rounds
    ]
    assert not (Path(explicit.run_dir) / "continual.json").exists()
    manifests = [
        json.loads((Path(result.run_dir) / "manifest.json").read_text(encoding="utf-8"))
        for result in (implicit, explicit)
    ]
    assert manifests[0]["scaler_parameters"] == manifests[1]["scaler_parameters"]


def test_training_schedule_cannot_select_outside_train_split(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kairots.runtime.worker as worker_module

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        save_checkpoints=False,
    ).to_dict()
    values["training"] = {"name": "Offline"}

    class LeakingSchedule:
        Config = get(category="training", name="Offline").Config

        def __init__(self, *, config: Any) -> None:
            self.config = config

        def select(self, **_: Any) -> list[tuple[int, int]]:
            return [(10**9, 10**9)]

        def commit(self, **_: Any) -> None:
            pass

    original_get = worker_module.get

    def selected_get(*, category: str, name: str) -> Any:
        if category == "training":
            return LeakingSchedule
        return original_get(category=category, name=name)

    monkeypatch.setattr(worker_module, "get", selected_get)
    with pytest.raises(RuntimeError, match="run failed") as error:
        run(experiment=Experiment.from_dict(values=values))
    assert "outside client 0's train split" in str(error.value)


def test_one_cycle_uses_the_batch_cap_and_logs_consumed_samples(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    experiment = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        strategy="Centralized",
        iterations=2,
    )
    values = experiment.to_dict()
    values["data"]["source"]["name"] = "ETTh1"
    values["data"]["source"]["batch_size"] = 4
    values["learner"].update(epochs=1, batches_per_epoch=2)
    values["scheduler"] = {"name": "OneCycle", "maximum_rate": 0.01}
    result = run(experiment=Experiment.from_dict(values=values))
    client_rows = [
        json.loads(line)
        for line in (Path(result.run_dir) / "clients.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(client_rows) == 2
    assert all(row["train_samples"] == 8 for row in client_rows)
    assert all(row["train_loss"] is not None for row in client_rows)


@pytest.mark.parametrize("strategy", ["FedAvg", "FLwF2"])
def test_class_incremental_name_preserves_continual_replay_and_parameters(
    workspace_tmp_path: Path, strategy: str
) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    values = configuration(root=workspace_tmp_path, raw=raw, strategy=strategy)
    values["training"].update(replay_size=2, capacity=4)
    old = run(experiment=Experiment.from_dict(values=values))
    renamed = deepcopy(values)
    renamed["training"]["name"] = "ClassIncremental"
    new = run(experiment=Experiment.from_dict(values=renamed))
    first = _parameters(directory=old.run_dir)
    last = _parameters(directory=new.run_dir)
    assert all(torch.equal(value, last[key]) for key, value in first.items())
    assert len(new.rounds) == 4
    assert [row["period"] for row in new.rounds] == [0, 0, 1, 1]
    reports = [
        json.loads(
            (Path(result.run_dir) / "continual.json").read_text(encoding="utf-8")
        )
        for result in (old, new)
    ]
    assert reports[0] == reports[1]


def test_class_incremental_refuses_forecasting_and_offline_allows_prequential() -> None:
    with pytest.raises(ValueError, match=r"task\.name"):
        Experiment.from_dict(
            values={
                "task": {"name": "Forecasting"},
                "training": {"name": "ClassIncremental"},
            }
        )
    with pytest.raises(ValueError, match=r"data\.source\.num_classes"):
        Experiment.from_dict(values={"training": {"name": "ClassIncremental"}})
    experiment = Experiment.from_dict(
        values={"training": {"name": "Offline"}, "evaluation": {"name": "Prequential"}}
    )
    assert experiment.training.name == "Offline"
