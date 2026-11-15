"""Configured optimizer choices resolve before running an experiment."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from kairots.__main__ import overlay
from kairots.config.schema import Experiment
from kairots.registry import get


def test_sgd_config_reaches_the_optimizer() -> None:
    experiment = Experiment.from_dict(
        values={
            "learner": {
                "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 0.0005},
                "fused_optimizer": False,
            }
        }
    )
    learner = get(category="learners", name=experiment.learner.name)(
        config=experiment.learner
    )
    optimizer = learner.optimizer(model=torch.nn.Linear(2, 1))
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.param_groups[0]["weight_decay"] == 0.0005


def test_worker_uses_the_configured_optimizer(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fixtures.synthetic import ett_experiment, write_ett_sources

    from kairots.learners.Gradient import Gradient
    from kairots.runtime.orchestrator import run

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", optimizer="SGD"
    ).to_dict()
    values["learner"]["optimizer"].update(momentum=0.9, weight_decay=0.0005)
    values["learner"]["fused_optimizer"] = False
    fit_epoch = Gradient.fit_epoch
    calls: list[int] = []

    def recorded(self: Gradient, **kwargs: Any) -> float:
        optimizer = kwargs["optimizer"]
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.9
        assert optimizer.param_groups[0]["weight_decay"] == 0.0005
        calls.append(1)
        return fit_epoch(self, **kwargs)

    monkeypatch.setattr(Gradient, "fit_epoch", recorded)
    run(experiment=Experiment.from_dict(values=values))
    assert calls


def test_optimizer_names_and_object_shape_fail_during_resolution() -> None:
    with pytest.raises(KeyError, match="Missing"):
        Experiment.from_dict(values={"learner": {"optimizer": {"name": "Missing"}}})
    with pytest.raises(TypeError, match="object"):
        Experiment.from_dict(values={"learner": {"optimizer": "SGD"}})


def test_cli_types_options_from_the_selected_nested_optimizer(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    config = overlay(
        path=str(path),
        arguments=[
            "--learner.optimizer.name",
            "SGD",
            "--learner.optimizer.momentum",
            "0.9",
        ],
    )
    assert config.learner.optimizer.name == "SGD"
    assert config.learner.optimizer.momentum == 0.9
