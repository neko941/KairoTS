from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.runtime.orchestrator import run


def test_zero_shot_trains_missing_source_and_reuses_only_complete_matching_runs(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kairots.runtime import orchestrator

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    source_values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "sources"
    ).to_dict()
    source_values["model"] = {"name": "DLinear", "individual": False}
    source_config = workspace_tmp_path / "source.json"
    source_config.write_text(json.dumps(source_values), encoding="utf-8")
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "targets", save_checkpoints=True
    ).to_dict()
    values["model"] = source_values["model"]
    values["strategy"] = {"name": "ZeroShot", "source_config": str(source_config)}
    values["initialization"] = {"name": "Pretrained"}
    trained: list[str] = []

    def train_source(*, experiment: Experiment) -> orchestrator.RunResult:
        assert experiment.strategy.name == "FedAvg"
        result = run(experiment=experiment)
        trained.append(result.run_dir)
        return result

    monkeypatch.setattr(orchestrator, "run", train_source)
    first = run(experiment=Experiment.from_dict(values=values))
    assert len(trained) == 1
    source = Path(trained[-1])
    assert (source / "checkpoints/final.pt").is_file()
    assert first.rounds[0]["selected_clients"] == []
    assert first.rounds[0]["communication_bytes"] == 0
    pinned = json.loads(
        (Path(first.run_dir) / "config.json").read_text(encoding="utf-8")
    )
    assert pinned["resolved"]["initialization"]["run_dir"] == str(source)
    second = run(experiment=Experiment.from_dict(values=values))
    assert len(trained) == 1
    assert first.rounds[0]["cross_generalization_test_pooled_mse"] == pytest.approx(
        second.rounds[0]["cross_generalization_test_pooled_mse"]
    )
    source_config.unlink()
    values["runtime"]["resume"] = first.run_dir
    from kairots.runtime.source_run import resolve_source

    resumed = resolve_source(experiment=Experiment.from_dict(values=values))
    assert len(trained) == 1 and resumed.initialization.run_dir == str(source)
    values["runtime"]["resume"] = ""
    values["initialization"]["run_dir"] = str(source)
    run(experiment=Experiment.from_dict(values=values))
    assert len(trained) == 1
    values["initialization"]["run_dir"] = ""
    source_config.write_text(json.dumps(source_values), encoding="utf-8")
    # A failed run with a checkpoint cannot satisfy the completed-run requirement.
    summary = source / "summary.json"
    summary.write_text('{"status": "failed"}', encoding="utf-8")
    run(experiment=Experiment.from_dict(values=values))
    assert len(trained) == 2
    # Changed training settings require a separate source run.
    source_values["learner"]["learning_rate"] *= 2
    source_config.write_text(json.dumps(source_values), encoding="utf-8")
    run(experiment=Experiment.from_dict(values=values))
    assert len(trained) == 3
    # Changed raw files invalidate the prepared-data keys even with the same config.
    path = raw / "ETTh1/ETTh1.csv"
    frame = pl.read_csv(path).with_columns((pl.col("OT") + 1).alias("OT"))
    frame.write_csv(path)
    run(experiment=Experiment.from_dict(values=values))
    assert len(trained) == 4
    # A damaged latest checkpoint is not reusable.
    (Path(trained[-1]) / "checkpoints/final.pt").write_bytes(b"incomplete")
    run(experiment=Experiment.from_dict(values=values))
    assert len(trained) == 5


def test_zero_shot_rejects_incompatible_source_before_training(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kairots.runtime import orchestrator

    source = workspace_tmp_path / "source.json"
    source.write_text(json.dumps({"model": {"name": "DLinear"}}), encoding="utf-8")
    values = {
        "model": {"name": "DLinear", "individual": True},
        "strategy": {"name": "ZeroShot", "source_config": str(source)},
        "initialization": {"name": "Pretrained"},
    }
    incompatible = Experiment.from_dict(values=values)
    with pytest.raises(ValueError, match="matching source and target"):
        run(experiment=incompatible)
    values["model"] = {"name": "DLinear", "moving_average_kernel": 3}

    def refuse_training(*, experiment: Experiment) -> orchestrator.RunResult:
        raise AssertionError("an incompatible source must not train")

    monkeypatch.setattr(orchestrator, "run", refuse_training)
    with pytest.raises(ValueError, match="matching source and target"):
        run(experiment=Experiment.from_dict(values=values))
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="experiment object"):
        run(experiment=Experiment.from_dict(values=values))


def test_zero_shot_resume_keeps_source_without_reloading_source_config(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    source_values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "sources"
    ).to_dict()
    source_values["model"] = {"name": "DLinear"}
    source_config = workspace_tmp_path / "source.json"
    source_config.write_text(json.dumps(source_values), encoding="utf-8")
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "targets", save_checkpoints=True
    ).to_dict()
    values["model"] = {"name": "DLinear"}
    values["strategy"] = {"name": "ZeroShot", "source_config": str(source_config)}
    values["initialization"] = {"name": "Pretrained"}
    unlink = Path.unlink

    def interrupted(
        self: Path, missing_ok: bool = False
    ) -> None:  # positional: pathlib callback.
        if self.name == "resume.pt" and "targets" in self.parts:
            raise KeyboardInterrupt
        unlink(self, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted)
        with pytest.raises(KeyboardInterrupt):
            run(experiment=Experiment.from_dict(values=values))
    directory = next((workspace_tmp_path / "targets").glob("*/*"))
    saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    source = saved["resolved"]["initialization"]["run_dir"]
    source_config.unlink()
    values["runtime"]["resume"] = str(directory)
    result = run(experiment=Experiment.from_dict(values=values))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pretraining_calendar_check"]["source_run"] == source
    assert result.rounds[0]["communication_bytes"] == 0
    assert len(list((workspace_tmp_path / "sources").glob("*/*/summary.json"))) == 1


def test_zero_shot_scores_without_training_and_requires_complete_checkpoint(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", save_checkpoints=True
    ).to_dict()
    values["model"] = {"name": "DLinear", "individual": False}
    source = run(experiment=Experiment.from_dict(values=values))
    values["strategy"] = {"name": "ZeroShot"}
    values["initialization"] = {"name": "Pretrained", "run_dir": source.run_dir}
    scored = run(experiment=Experiment.from_dict(values=values))
    row = scored.rounds[0]
    assert row["selected_clients"] == [] and row["communication_bytes"] == 0
    assert row["cross_generalization_test_pooled_mse"] == pytest.approx(
        source.rounds[-1]["generalization_test_pooled_mse"]
    )
    first = torch.load(Path(source.run_dir) / "checkpoints/final.pt", weights_only=True)
    last = torch.load(Path(scored.run_dir) / "checkpoints/final.pt", weights_only=True)
    assert all(torch.equal(value, last[key]) for key, value in first.items())
    manifest = json.loads((Path(source.run_dir) / "manifest.json").read_text())
    assert manifest["dataset_family"] == "ETDataset"
    exchange = workspace_tmp_path / "exchange"
    exchange.mkdir()
    pl.DataFrame(
        {
            "date": [date(1990, 1, 1) + timedelta(days=index) for index in range(240)],
            "0": [index / 100 for index in range(240)],
            "OT": [index / 80 for index in range(240)],
        }
    ).write_csv(exchange / "exchange_rate.csv")
    cross_values = Experiment.from_dict(values=values).to_dict()
    cross_values["data"]["source"] = {
        "name": "ExchangeRate",
        "raw_path": str(exchange),
        "download": False,
        "input_channels": ["0", "OT"],
    }
    cross_values["task"]["input_channels"] = []
    cross_values["model"].pop("input_channels", None)
    cross = run(experiment=Experiment.from_dict(values=cross_values))
    metadata = json.loads((Path(cross.run_dir) / "manifest.json").read_text())
    assert metadata["pretraining_calendar_check"]["cross_population"] == "skipped"
    assert cross.rounds[0]["communication_bytes"] == 0
    values["model"]["individual"] = True
    with pytest.raises(RuntimeError, match="matching source and target"):
        run(experiment=Experiment.from_dict(values=values))
    values["model"]["individual"] = False
    first.pop(next(iter(first)))
    torch.save(first, Path(source.run_dir) / "checkpoints/final.pt")
    with pytest.raises(RuntimeError, match="every checkpoint"):
        run(experiment=Experiment.from_dict(values=values))


def test_cross_family_skips_only_the_cross_population_calendar_check(
    workspace_tmp_path: Path,
) -> None:
    from kairots.initializers.Pretrained import Pretrained
    from kairots.models.DLinear import DLinear

    source = workspace_tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    model = DLinear(config=DLinear.Config())
    torch.save(model.state_dict(), source / "checkpoints/final.pt")
    client = {"name": "ETTh1", "cut_timestamps": {"validation": None, "test": 200}}
    (source / "manifest.json").write_text(
        json.dumps({"dataset_family": "ETDataset", "clients": [client]})
    )
    policy = Pretrained(config=Pretrained.Config(run_dir=str(source)))
    target = {
        "dataset_family": "ExchangeRate",
        "clients": [
            {
                "name": "exchange_rate",
                "cut_timestamps": {"validation": None, "test": 100},
            }
        ],
    }
    policy.apply(model=model, manifest=target)
    assert target["pretraining_calendar_check"]["cross_population"] == "skipped"
    target["dataset_family"] = "ETDataset"
    with pytest.raises(ValueError, match="past this run"):
        policy.apply(model=model, manifest=target)
    target["dataset_family"] = "ExchangeRate"
    target["clients"][0]["name"] = "ETTh1"
    with pytest.raises(ValueError, match="past this run"):
        policy.apply(model=model, manifest=target)
