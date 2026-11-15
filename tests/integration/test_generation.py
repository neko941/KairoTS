from __future__ import annotations

import math
from pathlib import Path

from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.runtime.orchestrator import run


def test_generation_trains_a_denoiser_and_scores_its_samples(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", iterations=2
    ).to_dict()
    values["task"] = {"name": "Generation", "diffusion_steps": 20}
    values["model"] = {"name": "Denoiser", "width": 16}
    values["evaluation"] = {"name": "Generative"}
    values["data"]["split"]["input_length"] = 0
    first, second = (
        run(experiment=Experiment.from_dict(values=values)).rounds[-1] for _ in range(2)
    )
    for name in (
        "mse",
        "mmd",
        "wasserstein",
        "acferror",
        "correlationerror",
        "discriminativescore",
        "tstr",
        "dcr",
    ):
        assert math.isfinite(first[f"generalization_test_pooled_{name}"]), name
    # Validation reports the denoising loss only.
    assert "generalization_val_pooled_mmd" not in first
    assert math.isfinite(first["generalization_val_pooled_mse"])
    # Noise and samples are seeded, so a rerun scores the same.
    assert {key: value for key, value in first.items() if "seconds" not in key} == {
        key: value for key, value in second.items() if "seconds" not in key
    }
