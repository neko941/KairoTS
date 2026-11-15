from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.runtime.orchestrator import run


def test_inverse_likelihood_rescales_standard_deviation_without_shifting_it(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs"
    ).to_dict()
    values["task"] = {"name": "LikelihoodForecasting"}
    values["model"] = {"name": "DLinear"}
    values.pop("evaluation")
    scaled = run(experiment=Experiment.from_dict(values=values))
    values["evaluation"] = {"inverse": True}
    inverse = run(experiment=Experiment.from_dict(values=values))
    manifest = json.loads((Path(scaled.run_dir) / "manifest.json").read_text())
    clients = manifest["clients"]
    count = sum(client["window_counts"]["test"] for client in clients)
    expected = (
        sum(
            float(torch.tensor(client["scaler"]["scale"]).log().mean())
            * client["window_counts"]["test"]
            for client in clients
        )
        / count
    )
    assert inverse.rounds[-1][
        "generalization_test_pooled_gaussiannll"
    ] == pytest.approx(
        scaled.rounds[-1]["generalization_test_pooled_gaussiannll"] + expected,
        abs=1e-6,
    )
