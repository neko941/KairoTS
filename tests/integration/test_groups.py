from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.runtime.orchestrator import run


def _values(*, raw: Path, runs: Path, model: dict[str, Any]) -> dict[str, Any]:
    values = ett_experiment(raw_path=raw, runs_dir=runs, iterations=2).to_dict()
    values["model"] = model
    return values


def test_groups_prepare_each_group_alone_and_score_across_them(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    runs = workspace_tmp_path / "runs"
    values = _values(raw=raw, runs=runs, model={"name": "DLinear"})
    alone = json.loads(
        (
            Path(run(experiment=Experiment.from_dict(values=values)).run_dir)
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    values["groups"] = [{}, {"data": {"source": {"input_channels": ["OT"]}}}]
    result = run(experiment=Experiment.from_dict(values=values))
    manifest = json.loads(
        (Path(result.run_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    count = len(alone["clients"])
    assert [entry["client_id"] for entry in manifest["clients"]] == list(
        range(2 * count)
    )
    assert [entry["group"] for entry in manifest["clients"]] == [0] * count + [
        1
    ] * count
    # Each group keeps its own cuts and train-fitted scalers: the first group's
    # equal the run without groups, the second scales OT alone with OT's values.
    first = manifest["clients"][:count]
    second = manifest["clients"][count:]
    assert [entry["scaler"] for entry in first] == [
        entry["scaler"] for entry in alone["clients"]
    ]
    assert [entry["scaler"]["mean"] for entry in second] == [
        entry["scaler"]["mean"][-1:] for entry in alone["clients"]
    ]
    # One server model (DLinear shares weights over channels): one cohort, no
    # personalization for FedAvg, a client mean across both groups.
    assert len(manifest["cohorts"]["generalization"]) == 1
    last = result.rounds[-1]
    assert "generalization_test_mean_client_mse" in last
    assert not any(key.startswith("personalization_") for key in last)


def test_groups_refuse_a_server_model_whose_shapes_differ(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    # One linear layer per channel: the OT-only group has fewer weights.
    values = _values(
        raw=raw,
        runs=workspace_tmp_path / "runs",
        model={"name": "DLinear", "individual": True},
    )
    values["groups"] = [{}, {"data": {"source": {"input_channels": ["OT"]}}}]
    with pytest.raises(RuntimeError, match="other parameter shapes"):
        run(experiment=Experiment.from_dict(values=values))


def test_groups_keep_their_clients_in_group_order(workspace_tmp_path: Path) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = _values(
        raw=raw, runs=workspace_tmp_path / "runs", model={"name": "DLinear"}
    )
    values["groups"] = [{"clients": [1]}, {"clients": [0]}]
    result = run(experiment=Experiment.from_dict(values=values))
    manifest = json.loads(
        (Path(result.run_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert [
        (entry["client_id"], entry["group"], entry["partition_client_id"])
        for entry in manifest["clients"]
    ] == [(0, 0, 1), (1, 1, 0)]


@pytest.mark.parametrize("strategy", ["FedAvg", "Centralized"])
def test_group_holdouts_never_train(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    from kairots.runtime.executors.Serial import Serial

    cached: set[int] = set()
    cache_client = Serial.cache_client

    def recorded(self: Serial, *, client_id: int, data: Any) -> None:
        cached.add(client_id)
        cache_client(self, client_id=client_id, data=data)

    monkeypatch.setattr(Serial, "cache_client", recorded)
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = _values(
        raw=raw, runs=workspace_tmp_path / "runs", model={"name": "DLinear"}
    )
    values["strategy"]["name"] = strategy
    values["groups"] = [
        {
            "data": {
                "split": {
                    "name": "Grouped",
                    "input_length": 8,
                    "output_length": 3,
                    "min_windows": 2,
                    "unseen_ratio": 0.5,
                }
            }
        },
        {},
    ]
    result = run(experiment=Experiment.from_dict(values=values))
    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text())
    unseen = set(manifest["unseen_clients"])
    assert unseen and unseen == set(manifest["groups"][0]["unseen_clients"])
    assert not manifest["groups"][1]["unseen_clients"]
    assert not cached.intersection(unseen)
    assert cached == {entry["client_id"] for entry in manifest["clients"]} - unseen
    for row in result.rounds:
        assert not unseen.intersection(row["selected_clients"])
        assert "unseen_generalization_test_pooled_mse" in row
