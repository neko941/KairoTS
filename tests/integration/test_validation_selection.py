"""Checkpoint selection follows the reported quantity's direction."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.evaluation.Frozen import Frozen
from kairots.runtime.orchestrator import run


@pytest.mark.parametrize("metric_enabled", [True, False])
def test_control_selects_the_highest_validation_return(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_enabled: bool,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        iterations=2,
        save_checkpoints=True,
    ).to_dict()
    values["task"] = {"name": "Control"}
    values["model"] = {"name": "Policy"}
    values["learner"] = {"name": "REINFORCE"}
    values["evaluation"] = {
        "metrics": {"validation": ["Return"] if metric_enabled else [], "test": []}
    }
    observed = []

    def evaluate(
        self: Frozen, *, phase: str, clients: list[Any], **kwargs: Any
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        if phase == "validation":
            observed.append(1)
        value = float(len(observed)) * (1 if metric_enabled else -1)
        return {"pooled_return": value}, [{"return": value} for _ in clients]

    monkeypatch.setattr(Frozen, "evaluate", evaluate)
    result = run(experiment=Experiment.from_dict(values=values))
    summary = json.loads((Path(result.run_dir) / "summary.json").read_text())
    assert len(observed) == 2
    assert summary["best_val_round"] == 1


def test_control_resume_refuses_a_checkpoint_using_the_old_direction(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kairots.runtime.executors.Serial import Serial

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        iterations=2,
        save_checkpoints=True,
    ).to_dict()
    values["task"] = {"name": "Control"}
    values["model"] = {"name": "Policy"}
    values["learner"] = {"name": "REINFORCE"}
    execute = Serial.execute
    values["evaluation"] = {}
    calls = []

    def interrupted(self: Serial, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return execute(self, **kwargs)

    monkeypatch.setattr(Serial, "execute", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(experiment=Experiment.from_dict(values=values))
    monkeypatch.undo()
    (run_dir,) = (workspace_tmp_path / "runs").glob("*/*")
    checkpoint = run_dir / "checkpoints/resume.pt"
    state = torch.load(checkpoint, weights_only=False)
    assert state.pop("selection_direction") == "max"
    torch.save(state, checkpoint)
    values["runtime"]["resume"] = str(run_dir)
    with pytest.raises(RuntimeError, match="validation direction"):
        run(experiment=Experiment.from_dict(values=values))
    state["selection_direction"] = "max"
    torch.save(state, checkpoint)
    assert len(run(experiment=Experiment.from_dict(values=values)).rounds) == 2


def test_patience_stops_after_rounds_without_a_better_validation_score(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", iterations=6
    ).to_dict()
    values["evaluation"] = {"patience": 2}
    scores = iter([1.0, 0.5, 0.6, 0.7, 0.1, 0.05])

    def evaluate(
        self: Frozen, *, phase: str, clients: list[Any], **kwargs: Any
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        value = next(scores) if phase == "validation" else 0.0
        return {"pooled_mse": value}, [{"mse": value} for _ in clients]

    monkeypatch.setattr(Frozen, "evaluate", evaluate)
    result = run(experiment=Experiment.from_dict(values=values))
    summary = json.loads((Path(result.run_dir) / "summary.json").read_text())
    assert summary["best_val_round"] == 1
    assert len(result.rounds) == 4
