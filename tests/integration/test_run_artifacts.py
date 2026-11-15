from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.registry import get
from kairots.runtime.orchestrator import run


def test_two_client_run_writes_complete_schema(workspace_tmp_path: Path) -> None:
    tmp_path = workspace_tmp_path
    raw = tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=tmp_path / "runs", save_checkpoints=True
    ).to_dict()
    values["initialization"] = {"name": "Average"}
    result = run(experiment=Experiment.from_dict(values=values))
    run_dir = Path(result.run_dir)
    assert {
        "config.json",
        "manifest.json",
        "rounds.jsonl",
        "clients.jsonl",
        "summary.json",
        "checkpoints",
    } <= {path.name for path in run_dir.iterdir()}
    round_row = json.loads((run_dir / "rounds.jsonl").read_text().splitlines()[0])
    assert isinstance(round_row["current_iter"], int)
    assert all(
        key in round_row
        for key in (
            "generalization_val_pooled_mse",
            "generalization_val_mean_client_mae",
            "generalization_test_pooled_mse",
            "generalization_test_mean_client_mae",
            "downlink_bytes",
            "uplink_bytes",
            "wall_seconds",
            "data_seconds",
            "fit_seconds",
            "eval_seconds",
            "aggregate_seconds",
            "transport_seconds",
        )
    )
    assert all(
        round_row[key] >= 0
        for key in (
            "data_seconds",
            "fit_seconds",
            "eval_seconds",
            "aggregate_seconds",
            "transport_seconds",
        )
    )
    client_rows = [
        json.loads(line)
        for line in (run_dir / "clients.jsonl").read_text().splitlines()
    ]
    assert all(
        {"data_seconds", "fit_seconds", "worker_seconds"} <= client["timing"].keys()
        for client in client_rows
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [client["client_id"] for client in manifest["clients"]] == [0, 1]
    assert len(manifest["raw_files"]) == 2
    assert manifest["initialization"]["policy"] == "Average"
    assert manifest["initialization"]["seed"] == 53
    assert manifest["initialization"]["parameters"]
    saved = json.loads((run_dir / "config.json").read_text())
    restored = Experiment.from_dict(values=saved["resolved"])
    model = get(category="models", name=restored.model.name)(config=restored.model)
    policy = get(category="initializers", name=restored.initialization.name)(
        config=restored.initialization
    )
    assert policy.apply(model=model) == manifest["initialization"]["parameters"]
    assert saved["seed"] == manifest["initialization"]["seed"]
    assert (run_dir / "checkpoints" / "final.pt").is_file()
    assert (run_dir / "checkpoints" / "best_val.pt").is_file()


@pytest.mark.parametrize("protocol", ["Frozen", "Adapted", "Prequential"])
def test_evaluation_protocols_share_a_result_contract(
    workspace_tmp_path: Path, protocol: str
) -> None:
    if protocol == "Prequential":
        raw = workspace_tmp_path / "raw"
        write_ett_sources(path=raw)
        values = ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / "runs",
            save_checkpoints=False,
        ).to_dict()
        values["evaluation"] = {"name": protocol, "steps": 0}
        values["data"]["view"] = {"name": "Ordered"}
        metric = "mse"
    else:
        from integration.test_classification import _config, _raw

        raw = _raw(root=workspace_tmp_path / "raw")
        values = _config(root=workspace_tmp_path, raw=raw, iterations=1)
        if protocol == "Adapted":
            values["model"] = {"name": "TimesNet"}
            values["evaluation"] = {"name": protocol, "steps": 0}
        else:
            values["evaluation"] = {"name": protocol}
        metric = "accuracy"
    result = run(experiment=Experiment.from_dict(values=values))
    round_row = result.rounds[-1]
    assert isinstance(round_row[f"generalization_test_pooled_{metric}"], float)
    if protocol != "Frozen":
        assert isinstance(
            round_row[f"generalization_test_frozen_pooled_{metric}"], float
        )
    rows = [
        json.loads(line)
        for line in (Path(result.run_dir) / "clients.jsonl").read_text().splitlines()
    ]
    assert rows and all(
        isinstance(row["generalization_test"][metric], float) for row in rows
    )


def test_too_short_client_is_recorded_and_excluded(workspace_tmp_path: Path) -> None:
    tmp_path = workspace_tmp_path
    raw = tmp_path / "raw"
    write_ett_sources(path=raw, short_etth1=True)
    result = run(
        experiment=ett_experiment(
            raw_path=raw, runs_dir=tmp_path / "runs", save_checkpoints=True
        )
    )
    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text())
    assert manifest["exclusions"][0]["client_id"] == 0
    assert "fewer than min_windows" in manifest["exclusions"][0]["exclusion_reason"]
    assert manifest["clients"][0]["client_id"] == 1


def test_worker_resolves_each_model_type_once_for_repeated_jobs(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kairots.runtime.worker as worker_module

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    experiment = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        iterations=2,
    )
    original_get = worker_module.get
    model_lookups: list[str] = []

    def track_get(*, category: str, name: str) -> Any:
        if category == "models":
            model_lookups.append(name)
        return original_get(category=category, name=name)

    monkeypatch.setattr(worker_module, "get", track_get)
    result = run(experiment=experiment)

    assert len(result.rounds) == 2
    assert model_lookups == [experiment.model.name]


def test_failed_run_still_writes_summary(workspace_tmp_path: Path) -> None:
    runs = workspace_tmp_path / "runs"
    experiment = ett_experiment(
        raw_path=workspace_tmp_path / "missing", runs_dir=runs, save_checkpoints=True
    )
    with pytest.raises(RuntimeError, match="run failed"):
        run(experiment=experiment)
    summaries = list(runs.rglob("summary.json"))
    assert len(summaries) == 1
    assert json.loads(summaries[0].read_text())["status"] == "failed"


def test_store_prepares_once_and_rekeys_on_change(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    tmp_path = workspace_tmp_path
    raw = tmp_path / "raw"
    write_ett_sources(path=raw)
    first = run(
        experiment=ett_experiment(
            raw_path=raw, runs_dir=tmp_path / "runs", save_checkpoints=True
        )
    )

    def no_csv(*args: object, **kwargs: object) -> None:  # positional: stub.
        raise AssertionError("a prepared store must not read raw CSV again")

    monkeypatch.setattr(pl, "read_csv", no_csv)
    second = run(
        experiment=ett_experiment(
            raw_path=raw, runs_dir=tmp_path / "runs", save_checkpoints=True
        )
    )
    assert [
        {key: value for key, value in row.items() if not key.endswith("_seconds")}
        for row in first.rounds
    ] == [
        {key: value for key, value in row.items() if not key.endswith("_seconds")}
        for row in second.rounds
    ]
    monkeypatch.undo()
    rekeyed = run(
        experiment=ett_experiment(
            raw_path=raw,
            runs_dir=tmp_path / "runs",
            min_windows=3,
            save_checkpoints=True,
        )
    )
    keys = [
        json.loads((Path(result.run_dir) / "manifest.json").read_text())["store"]
        for result in (first, second, rekeyed)
    ]
    assert keys[0] == keys[1]
    assert keys[2]["data_key"] == keys[0]["data_key"]
    assert keys[2]["split_key"] != keys[0]["split_key"]
    # The registry rebinds the class on the package, so patch the module itself.
    monkeypatch.setattr(
        sys.modules["kairots.data.stores.Parquet"],
        "code",
        lambda *, types: "changed code",
    )
    changed = run(
        experiment=ett_experiment(
            raw_path=raw, runs_dir=tmp_path / "runs", save_checkpoints=True
        )
    )
    store = json.loads((Path(changed.run_dir) / "manifest.json").read_text())["store"]
    assert store["data_key"] != keys[0]["data_key"]


def test_naive_stateless_run_trains_nothing_and_sends_nothing(
    workspace_tmp_path: Path,
) -> None:
    from kairots.config.schema import Experiment

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs"
    ).to_dict()
    values["model"] = {"name": "Naive"}
    values["learner"] = {"name": "Stateless"}
    result = run(experiment=Experiment.from_dict(values=values))
    assert all(row["communication_bytes"] == 0 for row in result.rounds)
    client_rows = (Path(result.run_dir) / "clients.jsonl").read_text().splitlines()
    assert all(json.loads(row)["train_loss"] == 0.0 for row in client_rows)


def test_jitter_touches_only_training_inputs(workspace_tmp_path: Path) -> None:
    from kairots.config.schema import Experiment

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)

    def test_mse(*, augmentation: dict[str, object] | None, name: str) -> float:
        values = ett_experiment(
            raw_path=raw, runs_dir=workspace_tmp_path / name
        ).to_dict()
        values["data"]["augmentation"] = augmentation
        result = run(experiment=Experiment.from_dict(values=values))
        return result.rounds[-1]["generalization_test_pooled_mse"]

    plain = test_mse(augmentation=None, name="plain")
    assert test_mse(augmentation={"name": "Jitter", "sigma": 0.0}, name="zero") == plain
    assert test_mse(augmentation={"name": "Jitter", "sigma": 1.0}, name="one") != plain


def test_cosine_scheduler_spans_the_run_or_each_round(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kairots.config.schema import Experiment
    from kairots.learners.Gradient import Gradient

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)

    def rounds(*, scheduler: str, span: str = "run") -> list[dict[str, object]]:
        values = ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / f"{scheduler}-{span}",
            iterations=2,
        ).to_dict()
        values["scheduler"] = {"name": scheduler}
        if scheduler == "Cosine":
            values["scheduler"]["span"] = span
        return run(experiment=Experiment.from_dict(values=values)).rounds

    constant, cosine = rounds(scheduler="Constant"), rounds(scheduler="Cosine")
    # Each one-epoch update of a round-span schedule starts at the full rate.
    starts: list[float] = []
    fit_epoch = Gradient.fit_epoch

    def recording(self: Gradient, **kwargs: Any) -> float:
        starts.append(kwargs["optimizer"].param_groups[0]["lr"])
        return fit_epoch(self, **kwargs)

    monkeypatch.setattr(Gradient, "fit_epoch", recording)
    per_round = rounds(scheduler="Cosine", span="round")
    assert len(starts) == 4 and len(set(starts)) == 1
    # A round-span schedule anneals within round 0 already.
    assert (
        constant[0]["generalization_val_pooled_mse"]
        != per_round[0]["generalization_val_pooled_mse"]
    )
    # Round 0 trains at the full rate under both; round 1 at half under Cosine.
    assert (
        constant[0]["generalization_val_pooled_mse"]
        == cosine[0]["generalization_val_pooled_mse"]
    )
    assert (
        constant[1]["generalization_val_pooled_mse"]
        != cosine[1]["generalization_val_pooled_mse"]
    )


def test_centralized_pools_only_the_seen_clients(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kairots.config.schema import Experiment
    from kairots.runtime.worker import Worker

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", iterations=1
    ).to_dict()
    values["data"]["split"].update({"name": "Grouped", "unseen_ratio": 0.5})
    values["strategy"]["name"] = "Centralized"
    pooled: list[int] = []
    update = Worker.update

    def recording(self: Worker, **kwargs: Any) -> Any:
        pooled.extend(kwargs["client_ids"])
        return update(self, **kwargs)

    monkeypatch.setattr(Worker, "update", recording)
    result = run(experiment=Experiment.from_dict(values=values))
    unseen = json.loads((Path(result.run_dir) / "manifest.json").read_text())[
        "unseen_clients"
    ]
    assert len(unseen) == 1 and pooled and unseen[0] not in pooled


def test_unseen_clients_never_train_and_score_as_their_own_population(
    workspace_tmp_path: Path,
) -> None:
    from kairots.config.schema import Experiment

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", iterations=2
    ).to_dict()
    values["data"]["split"].update({"name": "Grouped", "unseen_ratio": 0.5})
    result = run(experiment=Experiment.from_dict(values=values))
    run_dir = Path(result.run_dir)
    unseen = json.loads((run_dir / "manifest.json").read_text())["unseen_clients"]
    assert len(unseen) == 1
    assert all(unseen[0] not in row["selected_clients"] for row in result.rounds)
    assert all("unseen_generalization_test_pooled_mse" in row for row in result.rounds)
    rows = [
        json.loads(line)
        for line in (run_dir / "clients.jsonl").read_text().splitlines()
    ]
    held_out = [row for row in rows if row["client_id"] == unseen[0]]
    assert all(
        row["train_loss"] is None and "generalization_val" not in row
        for row in held_out
    )
    assert all("mse" in row["generalization_test"] for row in held_out)
