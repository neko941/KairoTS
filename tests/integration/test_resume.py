from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures.synthetic import ett_experiment, write_ett_sources
from torch import Tensor

from kairots.config.schema import Experiment
from kairots.runtime.contracts import Job, Result
from kairots.runtime.executors.Serial import Serial
from kairots.runtime.orchestrator import run


def _online_replay(
    *, raw: Path, runs_dir: Path, strategy: str, resume: str = ""
) -> Experiment:
    values = ett_experiment(
        raw_path=raw,
        runs_dir=runs_dir,
        strategy=strategy,
        iterations=3,
        save_checkpoints=True,
        optimizer="SGD" if strategy == "SCAFFOLD" else "Adam",
    ).to_dict()
    values["data"]["view"] = {"name": "Ordered"}
    values["evaluation"] = {
        "name": "Prequential",
        "regime": "Online",
        "replay_size": 4,
        "capacity": 8,
    }
    values["runtime"]["resume"] = resume
    return Experiment.from_dict(values=values)


def _rows(*, rounds: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: value for key, value in row.items() if not key.endswith("_seconds")}
        for row in rounds
    ]


# SCAFFOLD keeps a server control variate, restored from the checkpoint.
@pytest.mark.parametrize("strategy", ["FedAvg", "SCAFFOLD"])
def test_resumed_online_replay_run_matches_an_uninterrupted_one(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    whole = run(
        experiment=_online_replay(
            raw=raw, runs_dir=workspace_tmp_path / "whole", strategy=strategy
        )
    )
    assert not (Path(whole.run_dir) / "checkpoints" / "resume.pt").exists()

    calls = []
    execute = Serial.execute

    def interrupted(
        self: Serial, *, jobs: list[Job], broadcast: dict[str, Tensor]
    ) -> list[Result]:  # positional: self.
        calls.append(1)
        results = execute(self, jobs=jobs, broadcast=broadcast)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return results

    monkeypatch.setattr(Serial, "execute", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(
            experiment=_online_replay(
                raw=raw, runs_dir=workspace_tmp_path / "cut", strategy=strategy
            )
        )
    monkeypatch.undo()
    (run_dir,) = (workspace_tmp_path / "cut").glob("*/*")
    assert (run_dir / "checkpoints" / "resume.pt").is_file()
    # Another config, or data prepared differently, is refused before any write.
    original = (run_dir / "config.json").read_text(encoding="utf-8")
    changed = _online_replay(
        raw=raw, runs_dir=workspace_tmp_path / "cut", strategy=strategy
    ).to_dict()
    changed["runtime"]["resume"] = str(run_dir)
    changed["learner"]["learning_rate"] = 0.5
    with pytest.raises(ValueError, match="learner"):
        run(experiment=Experiment.from_dict(values=changed))
    assert (run_dir / "config.json").read_text(encoding="utf-8") == original
    manifest_path = run_dir / "manifest.json"
    saved = manifest_path.read_text(encoding="utf-8")
    tampered = json.loads(saved)
    tampered["raw_files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="prepared data changed"):
        run(
            experiment=_online_replay(
                raw=raw,
                runs_dir=workspace_tmp_path / "cut",
                strategy=strategy,
                resume=str(run_dir),
            )
        )
    manifest_path.write_text(saved, encoding="utf-8")
    resumed = run(
        experiment=_online_replay(
            raw=raw,
            runs_dir=workspace_tmp_path / "cut",
            strategy=strategy,
            resume=str(run_dir),
        )
    )
    assert _rows(rounds=resumed.rounds) == _rows(rounds=whole.rounds)
    lines = (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    whole_clients = (Path(whole.run_dir) / "clients.jsonl").read_text(encoding="utf-8")
    resumed_clients = (run_dir / "clients.jsonl").read_text(encoding="utf-8")
    assert [
        {key: value for key, value in json.loads(line).items() if key != "timing"}
        for line in resumed_clients.splitlines()
    ] == [
        {key: value for key, value in json.loads(line).items() if key != "timing"}
        for line in whole_clients.splitlines()
    ]
