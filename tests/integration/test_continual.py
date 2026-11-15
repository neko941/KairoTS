from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch
from torch import Tensor

from integration.test_classification import _config, _raw
from kairots.config.schema import Experiment
from kairots.data.stores.Parquet import Parquet
from kairots.runtime.contracts import Job, Result
from kairots.runtime.executors.Serial import Serial
from kairots.runtime.orchestrator import run
from kairots.training.ClassIncremental import ClassIncremental


def configuration(*, root: Path, raw: Path, strategy: str = "FLwF2") -> dict[str, Any]:
    values = _config(root=root, raw=raw, iterations=2)
    values["training"] = {"name": "ClassIncremental", "periods": 2}
    values["strategy"] = {"name": strategy, "iterations": 2}
    values["learner"].update(optimizer={"name": "SGD"}, learning_rate=0.01)
    values["runtime"]["save_checkpoints"] = True
    return values


def test_continual_rounds_keep_teachers_and_score_only_seen_tasks(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    execute = Serial.execute
    results: list[list[Result]] = []

    def recording(
        self: Serial, *, jobs: list[Job], broadcast: dict[str, Tensor]
    ) -> list[Result]:
        if results:
            previous = {result.operation_id[1]: result for result in results[-1]}
            for job in jobs:
                assert all(
                    torch.equal(value, job.state.extra["previous"][key])
                    for key, value in previous[job.client_id].uplink.tensors.items()
                )
        replies = execute(self, jobs=jobs, broadcast=broadcast)
        results.append(deepcopy(replies))
        return replies

    monkeypatch.setattr(Serial, "execute", recording)
    result = run(
        experiment=Experiment.from_dict(
            values=configuration(root=workspace_tmp_path, raw=raw)
        )
    )
    assert len(result.rounds) == 4
    assert "period_1_pooled_accuracy" not in result.rounds[1]
    assert "personalization_test_pooled_accuracy" in result.rounds[-1]
    assert "client_generalization_test_pooled_accuracy" in result.rounds[-1]
    matrix = json.loads((Path(result.run_dir) / "continual.json").read_text())
    assert [len(row) for row in matrix["R"]] == [1, 2]
    assert matrix["average_accuracy"] == pytest.approx(sum(matrix["R"][-1]) / 2)
    for row in result.rounds:
        assert row["downlink_bytes"] > 0
        assert row["uplink_bytes"] == row["downlink_bytes"] + 4 * len(
            row["selected_clients"]
        )


@pytest.mark.parametrize("replay_size", [0, 4])
def test_continual_resume_keeps_teachers_and_replay(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay_size: int,
) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    values = configuration(root=workspace_tmp_path, raw=raw)
    values["training"]["replay_size"] = replay_size
    whole = run(experiment=Experiment.from_dict(values=values))
    execute = Serial.execute

    def interrupted(
        self: Serial, *, jobs: list[Job], broadcast: dict[str, Tensor]
    ) -> list[Result]:
        if jobs[0].operation_id[0] == 3:
            raise KeyboardInterrupt
        return execute(self, jobs=jobs, broadcast=broadcast)

    monkeypatch.setattr(Serial, "execute", interrupted)
    values["runtime"]["runs_dir"] = str(workspace_tmp_path / "cut")
    with pytest.raises(KeyboardInterrupt):
        run(experiment=Experiment.from_dict(values=values))
    monkeypatch.undo()
    (directory,) = (workspace_tmp_path / "cut").glob("*/*")
    values["runtime"]["resume"] = str(directory)
    resumed = run(experiment=Experiment.from_dict(values=values))
    for actual, expected in zip(resumed.rounds, whole.rounds, strict=True):
        assert {k: v for k, v in actual.items() if not k.endswith("_seconds")} == {
            k: v for k, v in expected.items() if not k.endswith("_seconds")
        }


def test_future_task_values_do_not_change_earlier_training(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    execute = Serial.execute

    def interrupted(
        self: Serial, *, jobs: list[Job], broadcast: dict[str, Tensor]
    ) -> list[Result]:
        if jobs[0].operation_id[0] == 2:
            raise KeyboardInterrupt
        return execute(self, jobs=jobs, broadcast=broadcast)

    monkeypatch.setattr(Serial, "execute", interrupted)
    states = []
    for poisoned in (False, True):
        if poisoned:
            path = raw / "BasicMotions/BasicMotions_TRAIN.ts"
            lines = path.read_text().splitlines()
            path.write_text(
                "\n".join(
                    ":".join(
                        [",".join(["999"] * 100)] * 6
                        + [line.rsplit(":", maxsplit=1)[-1]]
                    )
                    if line.endswith((":Walking", ":Badminton"))
                    else line
                    for line in lines
                )
                + "\n"
            )
        values = configuration(root=workspace_tmp_path / str(poisoned), raw=raw)
        with pytest.raises(KeyboardInterrupt):
            run(experiment=Experiment.from_dict(values=values))
        (directory,) = (workspace_tmp_path / str(poisoned) / "runs").glob("*/*")
        states.append(
            torch.load(directory / "checkpoints/resume.pt", weights_only=False)[
                "global_params"
            ]
        )
    assert all(torch.equal(value, states[1][key]) for key, value in states[0].items())


def test_continual_serial_matches_ray(workspace_tmp_path: Path) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    results = []
    for executor in ("Serial", "Ray"):
        values = configuration(root=workspace_tmp_path / executor, raw=raw)
        values["runtime"]["name"] = executor
        values["training"]["replay_size"] = 4
        results.append(run(experiment=Experiment.from_dict(values=values)))
    for serial, distributed in zip(results[0].rounds, results[1].rounds, strict=True):
        for key, value in serial.items():
            if not key.endswith("_seconds"):
                assert distributed[key] == pytest.approx(value, rel=1e-5, abs=1e-6)


def test_future_round_values_do_not_change_earlier_training(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    config = Experiment.from_dict(
        values=configuration(
            root=workspace_tmp_path / "inspect", raw=raw, strategy="FedAvg"
        )
    )
    _, records = Parquet(config=config.data.store).prepare(experiment=config)
    regime = ClassIncremental(config=config.training)
    future: set[int] = set()
    present: set[int] = set()
    for record in records:
        _, splits = Parquet.load(record=record)
        samples = (
            pl.read_parquet(record.values_path)
            .unique(subset="sample", maintain_order=True)
            .get_column("sample")
            .to_list()
        )
        for current_iteration, selected in ((0, present), (1, future)):
            selected.update(
                samples[index]
                for index, _ in regime.released(
                    windows=splits["train"],
                    current_iteration=current_iteration,
                    iterations=2,
                    num_classes=4,
                )
            )
    assert future and not future.intersection(present)
    execute = Serial.execute

    def interrupted(
        self: Serial, *, jobs: list[Job], broadcast: dict[str, Tensor]
    ) -> list[Result]:
        if jobs[0].operation_id[0] == 1:
            raise KeyboardInterrupt
        return execute(self, jobs=jobs, broadcast=broadcast)

    monkeypatch.setattr(Serial, "execute", interrupted)
    states, scalers = [], []
    for poisoned in (False, True):
        if poisoned:
            path = raw / "BasicMotions/BasicMotions_TRAIN.ts"
            lines = path.read_text().splitlines()
            sample = -1
            for index, line in enumerate(lines):
                if line.startswith("@") or not line.strip():
                    continue
                sample += 1
                if sample in future:
                    lines[index] = ":".join(
                        [",".join(["999"] * 100)] * 6
                        + [line.rsplit(":", maxsplit=1)[1]]
                    )
            path.write_text("\n".join(lines) + "\n")
        root = workspace_tmp_path / str(poisoned)
        values = configuration(root=root, raw=raw, strategy="FedAvg")
        with pytest.raises(KeyboardInterrupt):
            run(experiment=Experiment.from_dict(values=values))
        (directory,) = (root / "runs").glob("*/*")
        states.append(
            torch.load(directory / "checkpoints/resume.pt", weights_only=False)[
                "global_params"
            ]
        )
        manifest = json.loads((directory / "manifest.json").read_text())
        scalers.append([client["scaler"] for client in manifest["clients"]])
    assert scalers[0] == scalers[1]
    assert all(torch.equal(value, states[1][key]) for key, value in states[0].items())
