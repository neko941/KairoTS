from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fixtures.synthetic import ett_experiment, write_ett_sources
from torch import Tensor

from kairots.config.schema import Experiment
from kairots.data.contracts import ClientRecord
from kairots.data.stores.Parquet import Parquet
from kairots.runtime.orchestrator import evaluate, fit, run


def test_fit_withholds_query_and_evaluate_uses_fit_artifact(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", save_checkpoints=False
    ).to_dict()
    values["model"] = {"name": "DLinear", "individual": False}
    experiment = Experiment.from_dict(values=values)

    loaded_records: list[ClientRecord] = []
    load = Parquet.load

    def load_fit_record(*, record: ClientRecord) -> tuple[Tensor, dict[str, list[Any]]]:
        if record.permissions is not None:
            loaded_records.append(record)
        return load(record=record)

    monkeypatch.setattr(Parquet, "load", staticmethod(load_fit_record))

    source = fit(experiment=experiment)
    assert loaded_records
    assert all(
        record.permissions == ("train", "validation") for record in loaded_records
    )
    assert all("test" not in record.windows for record in loaded_records)
    assert all(record.visible_until is not None for record in loaded_records)
    source_dir = Path(source.run_dir)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    assert manifest["data_permissions"] == {
        "mode": "fit",
        "source": ["train"],
        "support": ["validation"],
        "query": [],
        "query_labels": (
            "unavailable during fit; metric-only or post-prediction otherwise"
        ),
        "source_run": None,
    }
    source_round = json.loads((source_dir / "rounds.jsonl").read_text().splitlines()[0])
    assert not any("test" in name for name in source_round)
    assert (source_dir / "checkpoints" / "best_val.pt").is_file()

    result = evaluate(experiment=experiment, source_run=source.run_dir)
    target_dir = Path(result.run_dir)
    target_manifest = json.loads((target_dir / "manifest.json").read_text())
    assert target_manifest["data_permissions"]["mode"] == "evaluate"
    assert target_manifest["data_permissions"]["query"] == ["test"]
    target_round = json.loads((target_dir / "rounds.jsonl").read_text().splitlines()[0])
    assert target_round["selected_clients"] == []
    assert "cross_generalization_test_pooled_mse" in target_round
    assert not any("validation" in name for name in target_round)


def test_fit_without_validation_metrics_skips_selection_and_uses_final(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", save_checkpoints=True
    ).to_dict()
    values["model"] = {"name": "DLinear", "individual": False}
    values["evaluation"]["metrics"] = {"validation": [], "test": ["MSE"]}
    values["evaluation"]["selection_metric"] = ""
    experiment = Experiment.from_dict(values=values)

    source = fit(experiment=experiment)
    source_dir = Path(source.run_dir)
    summary = json.loads((source_dir / "summary.json").read_text())
    round_record = json.loads((source_dir / "rounds.jsonl").read_text().splitlines()[0])
    assert summary["best_val_round"] is None
    assert (source_dir / "checkpoints" / "final.pt").is_file()
    assert not (source_dir / "checkpoints" / "best_val.pt").exists()
    assert not any("validation" in key or "_val_" in key for key in round_record)

    result = evaluate(experiment=experiment, source_run=source.run_dir)
    target_round = json.loads(
        (Path(result.run_dir) / "rounds.jsonl").read_text().splitlines()[0]
    )
    assert "cross_generalization_test_pooled_mse" in target_round


def test_evaluate_rejects_artifact_not_created_by_fit(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    experiment = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", save_checkpoints=True
    )
    source = run(experiment=experiment)
    with pytest.raises(ValueError, match="produced by fit"):
        evaluate(experiment=experiment, source_run=source.run_dir)
