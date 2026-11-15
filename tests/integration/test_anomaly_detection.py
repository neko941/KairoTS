from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from kairots.config.schema import Experiment
from kairots.data.sources.SMD.SMDGroup1 import SMDGroup1
from kairots.runtime.orchestrator import run


def _raw(*, root: Path) -> Path:
    """Tiny SMD-shaped files: 38 channels; test spikes are labeled, except on the
    last machine, which has no anomalies (one-class test fold)."""
    generator = np.random.default_rng(0)
    time = np.arange(400)
    for index, machine in enumerate(SMDGroup1.MACHINES):
        base = np.stack([np.sin(time / 7 + channel) for channel in range(38)], axis=1)
        train = base + 0.05 * generator.standard_normal(base.shape)
        test = base + 0.05 * generator.standard_normal(base.shape)
        labels = np.zeros(400, dtype=int)
        if index < len(SMDGroup1.MACHINES) - 1:
            labels[200:210] = 1
            test[200:210] += 5.0
        for part, values, fmt in (
            ("train", train, "%.5f"),
            ("test", test, "%.5f"),
            ("test_label", labels[:, None], "%d"),
        ):
            path = root / part / f"{machine}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, values, delimiter=",", fmt=fmt)
    return root


def _config(*, root: Path) -> dict[str, Any]:
    config = json.loads(
        (
            Path(__file__).parents[2] / "configs/anomaly_detection/SMDGroup1.json"
        ).read_text(encoding="utf-8")
    )
    config["data"]["source"].update(
        raw_path=str(_raw(root=root / "raw")), download=False
    )
    config["data"]["source"]["batch_size"] = 16
    config["data"]["split"]["window_length"] = 20
    config["data"]["store"]["root"] = str(root / "datasets")
    config["strategy"]["iterations"] = 2
    config["learner"]["learning_rate"] = 0.01
    config["runtime"].update(
        device="cpu", runs_dir=str(root / "runs"), save_checkpoints=False
    )
    return config


def test_anomaly_run_detects_spikes_and_skips_one_class_client(
    workspace_tmp_path: Path,
) -> None:
    result = run(
        experiment=Experiment.from_dict(values=_config(root=workspace_tmp_path))
    )
    last = result.rounds[-1]
    assert last["generalization_test_pooled_adjustedf1"] > 0.5
    assert (
        last["generalization_test_pooled_f1"]
        <= last["generalization_test_pooled_adjustedf1"]
    )
    assert (
        "generalization_val_pooled_mse" in last
        and "generalization_val_pooled_f1" not in last
    )
    clients = [
        json.loads(line)
        for line in (Path(result.run_dir) / "clients.jsonl").read_text().splitlines()
    ]
    # The last machine has no anomalies: any alarm is a false positive, so F1 is 0.
    one_class = [row for row in clients if row["current_iter"] == 1][-1]
    assert one_class["generalization_test"]["f1"] == 0.0


def test_threshold_ignores_test_labels(workspace_tmp_path: Path) -> None:
    """Flipping every test label changes the scores, never which steps are flagged."""
    config = _config(root=workspace_tmp_path)
    first = run(experiment=Experiment.from_dict(values=config))
    for path in Path(config["data"]["source"]["raw_path"], "test_label").iterdir():
        labels = np.loadtxt(path, dtype=int)
        np.savetxt(path, 1 - labels, fmt="%d")
    config["runtime"]["runs_dir"] = str(workspace_tmp_path / "runs_flipped")
    second = run(experiment=Experiment.from_dict(values=config))
    # Same model and threshold: flagged steps were hits before, misses after.
    assert (
        first.rounds[-1]["generalization_test_pooled_mse"]
        == second.rounds[-1]["generalization_test_pooled_mse"]
    )
    assert (
        first.rounds[-1]["generalization_test_pooled_f1"]
        != second.rounds[-1]["generalization_test_pooled_f1"]
    )


@pytest.mark.parametrize(
    ("calibration", "leaks"),
    [("validation", False), ("train", False), ("train_test", True)],
)
def test_default_threshold_uses_no_test_data(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration: str,
    leaks: bool,
) -> None:
    """Poisoning every test value leaves the calibration scores unchanged by default."""
    quantile = torch.quantile
    calibrations: dict[str, list[torch.Tensor]] = {}
    for label in ("clean", "poisoned"):
        config = _config(root=workspace_tmp_path / label)
        config["evaluation"]["calibration"] = calibration
        if label == "poisoned":
            for path in Path(config["data"]["source"]["raw_path"], "test").iterdir():
                np.savetxt(path, np.loadtxt(path, delimiter=",") * 3 + 1, delimiter=",")
        recorded = calibrations.setdefault(label, [])

        def recording(values: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            recorded.append(values.clone())
            return quantile(values, *args, **kwargs)

        monkeypatch.setattr(torch, "quantile", recording)
        run(experiment=Experiment.from_dict(values=config))
    clean, poisoned = calibrations["clean"], calibrations["poisoned"]
    assert clean and len(clean) == len(poisoned)
    same = all(torch.equal(left, right) for left, right in zip(clean, poisoned))
    assert same != leaks
