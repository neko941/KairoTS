"""Native estimators select their fitting loop through their own requirements."""

import math
from pathlib import Path

import pytest
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.runtime.orchestrator import run


def test_xgboost_uses_native_fitting_when_the_learner_is_omitted(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", strategy="LocalOnly"
    ).to_dict()
    values.pop("learner")
    values["model"] = {
        "name": "XGBoost",
        "num_trees": 2,
        "max_depth": 2,
        "shrinkage": 0.17,
    }
    experiment = Experiment.from_dict(values=values)
    assert experiment.learner.name == "Estimator"
    result = run(experiment=experiment)
    assert len(result.rounds) == 1
    assert math.isfinite(result.rounds[0]["personalization_test_pooled_mse"])


def test_xgboost_rejects_an_explicit_gradient_learner() -> None:
    with pytest.raises(ValueError, match="XGBoost.*learner.name.*Estimator"):
        Experiment.from_dict(
            values={
                "model": {"name": "XGBoost"},
                "strategy": {"name": "LocalOnly"},
                "learner": {"name": "Gradient"},
            }
        )
