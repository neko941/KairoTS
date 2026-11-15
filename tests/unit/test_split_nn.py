from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.models.SplitNN import SplitNN
from kairots.runtime.orchestrator import run


def test_each_party_sees_only_its_own_history() -> None:
    torch.manual_seed(0)
    model = SplitNN(
        config=SplitNN.Config(input_length=8, output_length=3, input_channels=7)
    )
    batch_x = torch.randn(4, 11, 7)
    # Horizon rows hold NaN for targets and past covariates.
    batch_x[:, 8:] = torch.nan
    changed = batch_x.clone()
    changed[:, :8, 4:] += 1.0
    before = model.activations(batch_x=batch_x)
    after = model.activations(batch_x=changed)
    # Columns 4-6 belong to the second feature party only.
    torch.testing.assert_close(before[0], after[0])
    torch.testing.assert_close(before[1], after[1])
    assert not torch.allclose(before[2], after[2])
    assert model(batch_x=batch_x).isfinite().all()


def test_split_nn_pairs_only_with_its_strategy_and_bills_cut_traffic(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs"
    ).to_dict()
    values["task"] = {"name": "Forecasting", "target_channels": ["OT"]}
    values["model"] = {"name": "SplitNN", "embedding_dimension": 4}
    values["strategy"] = {"name": "FedAvg", "iterations": 1}
    with pytest.raises(ValueError, match="supports strategies"):
        Experiment.from_dict(values=values)
    values["strategy"] = {"name": "SplitNN", "iterations": 1}
    with pytest.raises(ValueError, match="supports models"):
        Experiment.from_dict(
            values={
                **values,
                "task": {"name": "Forecasting"},
                "model": {"name": "DLinear"},
            }
        )
    result = run(experiment=Experiment.from_dict(values=values))
    manifest = json.loads(
        (Path(result.run_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    windows = sum(client["window_counts"]["train"] for client in manifest["clients"])
    # Activations and gradients: 2 feature parties x 4 floats x 4 bytes per window.
    assert result.rounds[0]["uplink_bytes"] == 2 * windows * 2 * 4 * 4
    assert result.rounds[0]["downlink_bytes"] == 0
