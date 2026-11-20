"""Training must not depend on the test set: poisoning every test value (or test
sample) leaves every train/validation number and the trained parameters bitwise
identical, while the test scores change."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from integration.test_classification import _config as classification_config
from integration.test_classification import _raw as classification_raw
from integration.test_classification import _write_ts
from integration.test_m4 import _write_csvs as write_m4_csvs
from kairots.config.schema import Experiment
from kairots.data.sources.M4.M4Hourly import M4Hourly
from kairots.runtime.orchestrator import run


def _trained(*, result: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every non-test number of each round, and the saved final parameters."""
    rows = [
        {
            key: value
            for key, value in row.items()
            if "test_" not in key
            and not key.startswith("unseen_")
            and "seconds" not in key
        }
        for row in result.rounds
    ]
    checkpoints = Path(result.run_dir) / "checkpoints"
    states = {
        path.name: torch.load(path, weights_only=True)
        for path in sorted(checkpoints.glob("final*.pt"))
    }
    return rows, states


def _assert_same_training(*, clean: Any, poisoned: Any) -> None:
    clean_rows, clean_states = _trained(result=clean)
    poisoned_rows, poisoned_states = _trained(result=poisoned)
    assert clean_rows == poisoned_rows
    assert clean_states.keys() == poisoned_states.keys() and clean_states
    for name, state in clean_states.items():
        other = poisoned_states[name]
        # A model state, or one per client (`final_clients.pt`).
        if all(isinstance(value, torch.Tensor) for value in state.values()):
            state, other = {0: state}, {0: other}
        assert state.keys() == other.keys()
        state, other = list(state.values()), [other[key] for key in state]
        for left, right in zip(state, other, strict=True):
            for key, value in left.items():
                assert torch.equal(value, right[key]), (name, key)
    test_keys = [
        key for key in clean.rounds[-1] if key.startswith("generalization_test_pooled_")
    ]
    assert test_keys
    assert any(
        clean.rounds[-1][key] != poisoned.rounds[-1][key] for key in test_keys
    ), "the poison never reached the test scores"


@pytest.mark.parametrize(
    ("strategy", "protocol"),
    [
        ("FedAvg", "Frozen"),
        ("Ditto", "Frozen"),
        ("APFL", "Frozen"),
        ("FedAvg", "Prequential"),
        ("FedAvg", "TAFAS"),
        ("FedAvg", "Generative"),
        ("FedAvg", "CSDI"),
        ("FedAvg", "TimeGrad"),
    ],
)
def test_series_training_ignores_test_values(
    workspace_tmp_path: Path, strategy: str, protocol: str
) -> None:
    results = []
    for label in ("clean", "poisoned"):
        root = workspace_tmp_path / label
        write_ett_sources(path=root / "raw")
        experiment = ett_experiment(
            raw_path=root / "raw",
            runs_dir=root / "runs",
            strategy=strategy,
            iterations=2,
            save_checkpoints=True,
        )
        values = experiment.to_dict()
        values["evaluation"]["name"] = protocol
        if protocol == "Generative":
            # Synthesis: the denoiser trains on train windows only.
            values["task"] = {"name": "Generation", "diffusion_steps": 20}
            values["model"] = {"name": "Denoiser", "width": 16}
            values["evaluation"] = {"name": "Generative"}
            values["data"]["split"]["input_length"] = 0
        if protocol in ("CSDI", "TimeGrad"):
            values["task"] = {
                "name": "ProbabilisticImputation"
                if protocol == "CSDI"
                else "SampleForecasting"
            }
            values["model"] = {
                "name": protocol,
                "width": 8,
                "layers": 1,
                "diffusion_steps": 3,
                "samples": 4,
            }
            values["learner"] = {"name": "Denoising", "learning_rate": 0.001}
            values["evaluation"] = {"name": "Frozen"}
            values["data"]["split"]["stride"] = 12
            if protocol == "CSDI":
                values["data"]["split"]["input_length"] = 0
                values["model"]["heads"] = 2
            else:
                values["model"].update(lags=[1, 2], cells=4)
        if label == "poisoned":
            # Every value from each client's first test target on is replaced.
            manifest = json.loads(
                (Path(results[0].run_dir) / "manifest.json").read_text()
            )
            for client in manifest["clients"]:
                variant = client["name"].split("/")[-1]
                path = root / "raw" / variant / f"{variant}.csv"
                frame = pl.read_csv(path)
                cut = client["cut_timestamps"]["test"]
                frame = frame.with_columns(
                    pl.when(pl.col("date") >= cut)
                    .then(pl.exclude("date") * 1000 + 7)
                    .otherwise(pl.exclude("date"))
                )
                frame.write_csv(path)
        results.append(run(experiment=Experiment.from_dict(values=values)))
    _assert_same_training(clean=results[0], poisoned=results[1])


@pytest.mark.parametrize("protocol", ["Frozen", "Adapted"])
@pytest.mark.parametrize(
    "partition",
    ["IID", "Dirichlet", "ExtendedDirichlet", "Pathological", "ClassPerClient"],
)
def test_classification_training_ignores_test_samples(
    workspace_tmp_path: Path, protocol: str, partition: str
) -> None:
    results = []
    for label, seed in (("clean", 1), ("poisoned", 2)):
        root = workspace_tmp_path / label
        raw = classification_raw(root=root / "raw")
        # A different official TEST fold: new noise, every value shifted.
        _write_ts(
            path=raw / "BasicMotions" / "BasicMotions_TEST.ts",
            counts=(10, 10, 10, 10),
            seed=seed,
        )
        if label == "poisoned":
            path = raw / "BasicMotions" / "BasicMotions_TEST.ts"
            lines = path.read_text(encoding="utf-8").splitlines()
            lines = [
                line
                if line.startswith("@")
                else ":".join(
                    ",".join(
                        f"{float(value) * 50 + 3:.5f}" for value in part.split(",")
                    )
                    for part in line.split(":")[:-1]
                )
                + ":"
                + line.split(":")[-1]
                for line in lines
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        config = classification_config(root=root, raw=raw)
        config["data"]["partition"]["name"] = partition
        config["runtime"]["save_checkpoints"] = True
        if protocol == "Adapted":
            config["model"] = {"name": "TimesNet"}
            config["evaluation"] = {"name": "Adapted"}
        results.append(run(experiment=Experiment.from_dict(values=config)))
    _assert_same_training(clean=results[0], poisoned=results[1])


def test_m4_training_ignores_official_test_fold(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(M4Hourly, "SERIES_COUNT", 3)
    monkeypatch.setattr(M4Hourly, "HORIZON", 4)
    results = []
    for label in ("clean", "poisoned"):
        root = workspace_tmp_path / label
        write_m4_csvs(raw=root / "raw")
        if label == "poisoned":
            path = root / "raw" / "Test" / "Hourly-test.csv"
            header, *rows = path.read_text(encoding="utf-8").splitlines()
            rows = [
                ",".join(
                    [row.split(",")[0]]
                    + [str(float(value) * 9 + 1) for value in row.split(",")[1:]]
                )
                for row in rows
            ]
            path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        config = json.loads(
            (
                Path(__file__).parents[2] / "configs/examples/forecasting/M4Hourly.json"
            ).read_text(encoding="utf-8")
        )
        config["evaluation"]["metrics"]["test"].remove("M4OWA")
        config["data"]["source"].update(raw_path=str(root / "raw"), download=False)
        config["data"]["split"].update(input_length=8, output_length=4, train_stride=4)
        config["data"]["store"]["root"] = str(root / "datasets")
        config["strategy"]["iterations"] = 2
        config["runtime"].update(
            device="cpu", runs_dir=str(root / "runs"), save_checkpoints=True
        )
        results.append(run(experiment=Experiment.from_dict(values=config)))
    _assert_same_training(clean=results[0], poisoned=results[1])
