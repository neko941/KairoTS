from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.runtime.contracts import Message
from kairots.runtime.orchestrator import run
from kairots.strategies.Centralized import Centralized


def test_local_only_keeps_private_models_and_bills_nothing(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    result = run(
        experiment=ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / "runs",
            strategy="LocalOnly",
            iterations=2,
            save_checkpoints=True,
        )
    )
    assert all(row["communication_bytes"] == 0 for row in result.rounds)
    assert all(row["selected_clients"] == [0, 1] for row in result.rounds)
    client_models = torch.load(
        Path(result.run_dir) / "checkpoints" / "final_clients.pt", weights_only=True
    )
    assert len(client_models) == 2
    assert any(
        not torch.equal(client_models[0][key], client_models[1][key])
        for key in client_models[0]
    )


def test_local_only_xgboost_estimator_completes(workspace_tmp_path: Path) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        strategy="LocalOnly",
    ).to_dict()
    values["model"] = {"name": "XGBoost", "num_trees": 5, "max_depth": 2}
    values["learner"] = {"name": "Estimator"}
    result = run(experiment=Experiment.from_dict(values=values))
    assert result.rounds
    assert torch.isfinite(
        torch.tensor(result.rounds[-1]["generalization_test_pooled_mse"])
    )


def test_timesnet_forecasting_completes(workspace_tmp_path: Path) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        iterations=1,
    ).to_dict()
    values["model"] = {"name": "TimesNet"}
    result = run(experiment=Experiment.from_dict(values=values))
    assert result.rounds
    assert torch.isfinite(
        torch.tensor(result.rounds[-1]["generalization_test_pooled_mse"])
    )


def test_dfedavg_gossips_models_without_server_traffic(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    result = run(
        experiment=ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / "runs",
            strategy="DFedAvg",
            iterations=2,
            save_checkpoints=True,
        )
    )
    assert all(row["peer_bytes"] > 0 for row in result.rounds)
    assert all(row["downlink_bytes"] == 0 for row in result.rounds)
    assert all(row["uplink_bytes"] == 0 for row in result.rounds)
    client_models = torch.load(
        Path(result.run_dir) / "checkpoints" / "final_clients.pt", weights_only=True
    )
    assert len(client_models) == 2
    assert all(
        torch.allclose(client_models[0][key], client_models[1][key])
        for key in client_models[0]
    )


def test_centralized_trains_one_model_on_pooled_data(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    result = run(
        experiment=ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / "runs",
            strategy="Centralized",
            iterations=1,
        )
    )
    selected = result.rounds[0]["selected_clients"]
    assert len(selected) == 1
    assert result.rounds[0]["communication_bytes"] == 0
    client_rows = [
        json.loads(line)
        for line in (Path(result.run_dir) / "clients.jsonl").read_text().splitlines()
    ]
    row = next(row for row in client_rows if row["client_id"] == selected[0])
    assert row["train_samples"] > 0
    strategy = Centralized(config=Centralized.Config())
    stub = Message(sender="server", receiver="server")
    with pytest.raises(ValueError):
        strategy.aggregate(uplinks=[stub, stub], global_params={})


def test_dfedprox_anchors_on_each_clients_own_start_model(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    result = run(
        experiment=ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / "runs",
            strategy="DFedProx",
            iterations=2,
        )
    )
    assert all(row["peer_bytes"] > 0 for row in result.rounds)
    assert all(row["downlink_bytes"] == 0 for row in result.rounds)


def test_fedpmt_uploads_only_trained_groups(workspace_tmp_path: Path) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        strategy="FedPMT",
        iterations=2,
    ).to_dict()
    values["model"] = {"name": "DLinear"}
    result = run(experiment=Experiment.from_dict(values=values))
    # DLinear has two equal groups (seasonal, trend): client 0 uploads the trend
    # only, client 1 both, so the uplink is three quarters of the downlink.
    assert all(
        row["uplink_bytes"] * 4 == row["downlink_bytes"] * 3 for row in result.rounds
    )


def test_fedpaq_bills_the_quantized_uplink(workspace_tmp_path: Path) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    result = run(
        experiment=ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / "runs",
            strategy="FedPAQ",
            iterations=2,
        )
    )
    # Default 8 levels: a sign bit and 4 magnitude bits per value plus a 32-bit
    # norm, per client.
    for row in result.rounds:
        values_per_client = row["downlink_bytes"] // 4 // 2
        assert row["uplink_bytes"] == 2 * -(-(values_per_client * 5 + 32) // 8)


def test_messages_bill_only_what_crosses_the_network(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)

    def traffic(*, strategy: str) -> tuple[int, int, int]:
        # SGD for every strategy, as SCAFFOLD requires; optimizer state is not billed.
        values = ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / "runs",
            strategy=strategy,
            optimizer="SGD",
        ).to_dict()
        result = run(experiment=Experiment.from_dict(values=values))
        row = result.rounds[0]
        client_rows = [
            json.loads(line)
            for line in (Path(result.run_dir) / "clients.jsonl")
            .read_text()
            .splitlines()
            if json.loads(line)["current_iter"] == row["current_iter"]
        ]
        # Every billed byte of this setting is attributed to one client.
        for kind in ("downlink_bytes", "uplink_bytes", "peer_bytes"):
            assert sum(client_row[kind] for client_row in client_rows) == row[kind]
        return row["downlink_bytes"], row["uplink_bytes"], len(row["selected_clients"])

    downlink, uplink, clients = traffic(strategy="FedAvg")
    # Sample-weighted averaging: the model plus a 4-byte sample count per client.
    assert uplink == downlink + 4 * clients and downlink > 0
    # Personal models stay on the client; APFL and SCAFFOLD average unweighted, and
    # control variates double both directions.
    assert traffic(strategy="Ditto") == (downlink, uplink, clients)
    assert traffic(strategy="APFL") == (downlink, downlink, clients)
    assert traffic(strategy="SCAFFOLD") == (2 * downlink, 2 * downlink, clients)
