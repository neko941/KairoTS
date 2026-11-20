from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kairots.config.schema import Experiment
from kairots.data.sources.UEA import UEA
from kairots.data.stores.Parquet import Parquet
from kairots.evaluation.Frozen import Frozen
from kairots.registry import get
from kairots.runtime.orchestrator import run

CLASSES = ("Standing", "Running", "Walking", "Badminton")


def _write_ts(*, path: Path, counts: tuple[int, ...], seed: int) -> None:
    """A BasicMotions-shaped `.ts` file: 6 dimensions, length 100, `counts[k]` samples
    of class k."""
    generator = np.random.default_rng(seed)
    lines = [
        "@problemName BasicMotions",
        "@univariate false",
        "@dimensions 6",
        "@equalLength true",
        "@seriesLength 100",
        f"@classLabel true {' '.join(CLASSES)}",
        "@data",
    ]
    time = np.arange(100)
    for label, (name, count) in enumerate(zip(CLASSES, counts, strict=True)):
        for _ in range(count):
            dimensions = [
                np.sin(time * (label + 1) / 10 + dimension)
                + 0.1 * generator.standard_normal(100)
                for dimension in range(6)
            ]
            lines.append(
                ":".join(
                    ",".join(f"{value:.5f}" for value in row) for row in dimensions
                )
                + f":{name}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _raw(*, root: Path, train_counts: tuple[int, ...] = (10, 10, 10, 10)) -> Path:
    for fold, counts, seed in (
        ("TRAIN", train_counts, 0),
        ("TEST", (10, 10, 10, 10), 1),
    ):
        _write_ts(
            path=root / "BasicMotions" / f"BasicMotions_{fold}.ts",
            counts=counts,
            seed=seed,
        )
    return root


def _config(*, root: Path, raw: Path, iterations: int = 2) -> dict[str, Any]:
    config = json.loads(
        (
            Path(__file__).parents[2]
            / "configs/examples/classification/BasicMotions.json"
        ).read_text(encoding="utf-8")
    )
    config["data"]["source"].update(raw_path=str(raw), download=False)
    config["data"]["store"]["root"] = str(root / "datasets")
    config["strategy"]["iterations"] = iterations
    config["runtime"].update(
        device="cpu", runs_dir=str(root / "runs"), save_checkpoints=False
    )
    return config


def test_ts_reader_round_trips_labels_and_shapes(workspace_tmp_path: Path) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    classes, samples, labels = UEA.read_ts(
        path=raw / "BasicMotions" / "BasicMotions_TRAIN.ts"
    )
    assert classes == list(CLASSES)
    assert len(samples) == 40 and len(samples[0]) == 6 and len(samples[0][0]) == 100
    assert labels.count("Running") == 10


def test_classification_run_scores_accuracy(workspace_tmp_path: Path) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    config = _config(root=workspace_tmp_path, raw=raw)
    result = run(experiment=Experiment.from_dict(values=config))
    last = result.rounds[-1]
    assert 0.0 <= last["generalization_test_pooled_accuracy"] <= 1.0
    # Validation is off by default: no val metrics, no best-round pick.
    assert not any(key.startswith("generalization_val_") for key in last)
    summary = json.loads((Path(result.run_dir) / "summary.json").read_text())
    assert summary["best_val_round"] is None
    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text())
    assert manifest["classes"] == list(CLASSES)
    assert sum(client["window_counts"]["test"] for client in manifest["clients"]) == 40


def test_fml_trains_a_private_model_per_cohort(workspace_tmp_path: Path) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    config = _config(root=workspace_tmp_path, raw=raw)
    config["data"]["partition"] = {"name": "IID", "num_clients": 4, "seed": 0}
    config["strategy"] = {"name": "FML", "iterations": 2}
    config["model"] = {"name": "DLinear"}
    config["groups"] = [
        {
            "clients": [0, 1],
            "client_model": {
                "name": "TimesNet",
                "d_model": 4,
                "d_ff": 8,
                "encoder_layers": 1,
                "k": 2,
                "num_inception_kernels": 1,
            },
        },
        {"clients": [2, 3], "client_model": {"name": "DLinear"}},
    ]
    result = run(experiment=Experiment.from_dict(values=config))
    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text())
    assert len(manifest["cohorts"]["personalization"]) == 2
    last = result.rounds[-1]
    for key in (
        "personalization_test_pooled_accuracy",
        "personalization_cohort0_test_pooled_accuracy",
        "personalization_cohort1_test_pooled_accuracy",
        "generalization_test_pooled_accuracy",
    ):
        assert key in last
    # FML averages unweighted, so no sample count is sent.
    assert all(row["uplink_bytes"] == row["downlink_bytes"] for row in result.rounds)


def test_fedbabu_shares_the_body_and_scores_after_fine_tuning(
    workspace_tmp_path: Path,
) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    rounds = {}
    for name in ("FedBABU", "FedAvg"):
        config = _config(root=workspace_tmp_path / name, raw=raw)
        config["strategy"]["name"] = name
        config["evaluation"]["finetune_epochs"] = 1
        rounds[name] = run(experiment=Experiment.from_dict(values=config)).rounds
    babu, average = rounds["FedBABU"], rounds["FedAvg"]
    assert "finetuned_test_pooled_accuracy" not in babu[0]
    assert 0.0 <= babu[-1]["finetuned_test_pooled_accuracy"] <= 1.0
    assert "finetuned_test_pooled_accuracy" in average[-1]
    assert not any(key.startswith("personalization_") for key in babu[-1])
    for row, reference in zip(babu, average, strict=True):
        # The body plus the 4-byte sample count that weights its average.
        assert row["uplink_bytes"] == row["downlink_bytes"] + 4 * len(
            row["selected_clients"]
        )
        assert row["uplink_bytes"] < reference["uplink_bytes"]


def test_dirichlet_identity_classification_run_trains_both_clients(
    workspace_tmp_path: Path,
) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    config = _config(root=workspace_tmp_path, raw=raw)
    config["data"]["partition"] = {"name": "Dirichlet", "num_clients": 2}
    config["data"]["scaler"] = {"name": "Identity"}
    result = run(experiment=Experiment.from_dict(values=config))
    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text())
    assert [client["client_id"] for client in manifest["clients"]] == [0, 1]


def test_client_missing_a_class_still_trains_and_scores(
    workspace_tmp_path: Path,
) -> None:
    # One Badminton train sample: exactly one of the two IID clients ever sees it.
    raw = _raw(root=workspace_tmp_path / "raw", train_counts=(10, 10, 10, 1))
    result = run(
        experiment=Experiment.from_dict(
            values=_config(root=workspace_tmp_path, raw=raw)
        )
    )
    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text())
    assert [client["client_id"] for client in manifest["clients"]] == [0, 1]
    assert 0.0 <= result.rounds[-1]["generalization_test_pooled_macrof1"] <= 1.0


def test_checkpoint_round_trip_reproduces_test_scores(
    workspace_tmp_path: Path,
) -> None:
    raw = _raw(root=workspace_tmp_path / "raw")
    config = _config(root=workspace_tmp_path, raw=raw)
    config["runtime"]["save_checkpoints"] = True
    experiment = Experiment.from_dict(values=config)
    result = run(experiment=experiment)
    model = get(category="models", name=experiment.model.name)(config=experiment.model)
    model.load_state_dict(torch.load(Path(result.run_dir) / "checkpoints" / "final.pt"))
    _, records = Parquet(config=experiment.data.store).prepare(experiment=experiment)
    clients = []
    for record in records:
        clients.append(Parquet.load(record=record))
    loss_type = get(category="metrics", name=experiment.evaluation.loss)
    metrics, _ = Frozen(
        config=experiment.evaluation, loss_function=loss_type(config=loss_type.Config())
    ).evaluate(
        phase="test",
        model=model,
        clients=clients,
        experiment=experiment,
        view=get(category="data/views", name=experiment.data.view.name)(
            config=experiment.data.view
        ),
        task=get(category="tasks", name=experiment.task.name)(config=experiment.task),
        device=torch.device("cpu"),
    )
    last = result.rounds[-1]
    for key in ("pooled_crossentropy", "pooled_accuracy", "pooled_macrof1"):
        assert abs(metrics[key] - last[f"generalization_test_{key}"]) < 1e-6, key


def _adapted_setup(
    *, workspace_tmp_path: Path, steps: int
) -> tuple[Any, Any, dict[str, Any]]:
    """An untrained TimesNet, the TENT protocol and its evaluate arguments."""
    raw = _raw(root=workspace_tmp_path / "raw")
    config = _config(root=workspace_tmp_path, raw=raw)
    config["model"] = {"name": "TimesNet"}
    config["evaluation"] = {"name": "Adapted", "adaptation": "TENT", "steps": steps}
    experiment = Experiment.from_dict(values=config)
    torch.manual_seed(0)
    model = get(category="models", name="TimesNet")(config=experiment.model)
    _, records = Parquet(config=experiment.data.store).prepare(experiment=experiment)
    loss_type = get(category="metrics", name=experiment.evaluation.loss)
    protocol = get(category="evaluation", name="Adapted")(
        config=experiment.evaluation, loss_function=loss_type(config=loss_type.Config())
    )
    arguments = {
        "phase": "test",
        "model": model,
        "clients": [Parquet.load(record=record) for record in records],
        "experiment": experiment,
        "view": get(category="data/views", name=experiment.data.view.name)(
            config=experiment.data.view
        ),
        "task": get(category="tasks", name=experiment.task.name)(
            config=experiment.task
        ),
        "device": torch.device("cpu"),
    }
    return model, protocol, arguments


def test_zero_adaptation_steps_reproduce_frozen_scores(
    workspace_tmp_path: Path,
) -> None:
    _, protocol, arguments = _adapted_setup(
        workspace_tmp_path=workspace_tmp_path, steps=0
    )
    metrics, _ = protocol.evaluate(**arguments)
    assert metrics["adaptation_updates"] == 0.0
    for key in ("pooled_crossentropy", "pooled_accuracy"):
        assert metrics[key] == metrics[f"frozen_{key}"], key


def test_tent_updates_keep_source_model_and_ignore_labels(
    workspace_tmp_path: Path,
) -> None:
    model, protocol, arguments = _adapted_setup(
        workspace_tmp_path=workspace_tmp_path, steps=1
    )
    source = {key: value.clone() for key, value in model.state_dict().items()}
    recorded: list[torch.Tensor] = []
    predict = protocol.predict

    def recording(**kwargs: Any) -> torch.Tensor:
        prediction = predict(**kwargs)
        if protocol.adapting:
            recorded.append(prediction)
        return prediction

    protocol.predict = recording
    metrics, _ = protocol.evaluate(**arguments)
    # 40 test samples over 2 clients, batches of 16: 2 batches per client.
    assert metrics["adaptation_updates"] == 4.0
    assert metrics["pooled_crossentropy"] != metrics["frozen_pooled_crossentropy"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, source[key]), key
    # Hidden labels changed, inputs fixed: identical adapted predictions.
    first, recorded[:] = list(recorded), []
    task = arguments["task"]
    prepare = task.prepare

    def relabeled(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch_x, batch_y = prepare(**kwargs)
        return batch_x, (batch_y + 1) % len(CLASSES)

    task.prepare = relabeled
    protocol.evaluate(**arguments)
    assert len(recorded) == len(first)
    for left, right in zip(first, recorded, strict=True):
        assert torch.equal(left, right)
