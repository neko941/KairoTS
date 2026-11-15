"""The historical TEST-selected protocol is gated; the clean one never reads TEST.

Reference rules are the pinned TSLib classification experiment: `EarlyStopping`
on the score `-(-accuracy)` (a tie is an improvement), the `lradj="type1"` rate,
the gradient-norm clip of 4 and `UEAloader.instance_norm`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from integration.test_classification import _config as classification_config
from integration.test_classification import _raw as classification_raw
from integration.test_classification import _write_ts
from integration.test_leakage import _assert_same_training
from kairots.config.schema import Experiment
from kairots.data.views.InstanceNormalized import InstanceNormalized
from kairots.evaluation.Frozen import Frozen
from kairots.learners.Gradient import Gradient
from kairots.runtime.orchestrator import run
from kairots.schedulers.EpochHalving import EpochHalving

ROOT = Path(__file__).parents[2]


def _author_early_stopping(
    *, accuracies: list[float], patience: int
) -> tuple[int, int]:
    """TSLib `EarlyStopping` fed `-accuracy`: (kept checkpoint, epochs run)."""
    best_score, counter, kept = None, 0, -1
    for index, accuracy in enumerate(accuracies):
        score = -(-accuracy)
        if best_score is None or not score < best_score:
            best_score, counter, kept = score, 0, index
        else:
            counter += 1
            if counter >= patience:
                return kept, index + 1
    return kept, len(accuracies)


def _scripted(
    *, monkeypatch: pytest.MonkeyPatch, validation: list[float], test: list[float]
) -> None:
    calls = {"validation": 0, "test": 0}
    scripts = {"validation": validation, "test": test}

    def evaluate(
        self: Frozen, *, phase: str, clients: list[Any], **kwargs: Any
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        value = scripts[phase][calls[phase]]
        calls[phase] += 1
        metrics = {"pooled_accuracy": value, "pooled_crossentropy": 1.0}
        return metrics, [{"accuracy": value, "crossentropy": 1.0} for _ in clients]

    monkeypatch.setattr(Frozen, "evaluate", evaluate)


def _scripted_run(
    *, root: Path, leaky: bool, patience: int, iterations: int
) -> dict[str, Any]:
    raw = classification_raw(root=root / "raw")
    config = classification_config(root=root, raw=raw, iterations=iterations)
    config["evaluation"] = {
        "patience": patience,
        "selection_metric": "Accuracy",
        "leaky_test_selection": leaky,
        "metrics": {
            "validation": [] if leaky else ["Accuracy"],
            "test": ["Accuracy"],
        },
    }
    if not leaky:
        config["data"]["split"]["validation_ratio"] = 0.25
    result = run(experiment=Experiment.from_dict(values=config))
    return {
        "summary": json.loads((Path(result.run_dir) / "summary.json").read_text()),
        "rounds": len(result.rounds),
    }


def test_test_selection_picks_and_stops_like_the_pinned_loop(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accuracies = [0.30, 0.50, 0.50, 0.45, 0.60, 0.60, 0.20, 0.10, 0.10, 0.10]
    for patience in (2, 3):
        _scripted(monkeypatch=monkeypatch, validation=[], test=accuracies)
        observed = _scripted_run(
            root=workspace_tmp_path / str(patience),
            leaky=True,
            patience=patience,
            iterations=10,
        )
        kept, epochs = _author_early_stopping(accuracies=accuracies, patience=patience)
        # The later of two equal best scores is kept, and a tie resets patience.
        assert kept == 5
        assert observed["summary"]["best_val_round"] == kept
        assert observed["rounds"] == epochs
        assert observed["summary"]["selection"] == {
            "phase": "test",
            "metric": "Accuracy",
            "leaky_test_selection": True,
        }


def test_clean_selection_ignores_the_test_scores(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted(
        monkeypatch=monkeypatch,
        validation=[0.9, 0.5, 0.4, 0.3],
        test=[0.1, 0.2, 0.3, 0.99],
    )
    observed = _scripted_run(
        root=workspace_tmp_path, leaky=False, patience=0, iterations=4
    )
    assert observed["summary"]["best_val_round"] == 0
    assert observed["summary"]["selection"]["leaky_test_selection"] is False


def test_the_test_selected_protocol_is_never_a_default() -> None:
    assert Frozen.Config().leaky_test_selection is False
    flagged = {
        path.name
        for path in (ROOT / "configs").glob("*/*.json")
        if json.loads(path.read_text(encoding="utf-8"))
        .get("evaluation", {})
        .get("leaky_test_selection")
    }
    assert flagged == {"EthanolConcentration-TimesNet-TestSelected.json"}


def test_the_test_selected_protocol_refuses_a_validation_fold(
    workspace_tmp_path: Path,
) -> None:
    raw = classification_raw(root=workspace_tmp_path / "raw")
    config = classification_config(root=workspace_tmp_path, raw=raw)
    config["data"]["split"]["validation_ratio"] = 0.25
    config["evaluation"] = {"leaky_test_selection": True}
    with pytest.raises(RuntimeError, match="no validation fold"):
        run(experiment=Experiment.from_dict(values=config))


def test_a_selection_metric_must_be_reported_in_its_phase(
    workspace_tmp_path: Path,
) -> None:
    raw = classification_raw(root=workspace_tmp_path / "raw")
    config = classification_config(root=workspace_tmp_path, raw=raw)
    config["evaluation"] = {
        "selection_metric": "Accuracy",
        "metrics": {"validation": [], "test": ["Accuracy"]},
    }
    with pytest.raises(ValueError, match="must list 'Accuracy'"):
        run(experiment=Experiment.from_dict(values=config))


def test_clean_track_training_and_selection_ignore_test_samples(
    workspace_tmp_path: Path,
) -> None:
    results = []
    for label, seed in (("clean", 1), ("poisoned", 2)):
        root = workspace_tmp_path / label
        raw = classification_raw(root=root / "raw")
        path = raw / "BasicMotions" / "BasicMotions_TEST.ts"
        _write_ts(path=path, counts=(10, 10, 10, 10), seed=seed)
        if label == "poisoned":
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
        config = classification_config(root=root, raw=raw, iterations=4)
        config["runtime"]["save_checkpoints"] = True
        config["data"]["split"]["validation_ratio"] = 0.25
        config["evaluation"] = {
            "patience": 3,
            "selection_metric": "Accuracy",
            "metrics": {"validation": ["Accuracy"], "test": ["Accuracy"]},
        }
        results.append(run(experiment=Experiment.from_dict(values=config)))
    _assert_same_training(clean=results[0], poisoned=results[1])
    summaries = [
        json.loads((Path(result.run_dir) / "summary.json").read_text())
        for result in results
    ]
    assert summaries[0]["best_val_round"] == summaries[1]["best_val_round"]
    assert summaries[0]["selection"]["phase"] == "validation"
    # The checkpoint chosen on validation is the same in both runs.
    first, second = (
        torch.load(Path(result.run_dir) / "checkpoints" / "best_val.pt")
        for result in results
    )
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_instance_normalization_is_the_author_loader_rule() -> None:
    generator = torch.Generator().manual_seed(0)
    batch = torch.randn(5, 40, 3, generator=generator) * 7 + 100

    def author(case: torch.Tensor) -> torch.Tensor:
        mean = case.mean(0, keepdim=True)
        case = case - mean
        stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
        case /= stdev
        return case

    view = InstanceNormalized(config=InstanceNormalized.Config())
    (observed, _), *rest = view.batches(
        values=batch,
        windows=[(index, index) for index in range(5)],
        input_length=40,
        output_length=0,
        offset_length=0,
        batch_size=5,
        shuffle=False,
        generator=generator,
    )
    assert not rest
    assert torch.allclose(
        observed, torch.stack([author(case.clone()) for case in batch]), atol=1e-5
    )


def test_epoch_halving_follows_tslib_type1_across_resumed_updates() -> None:
    # The author sets `lr = 0.001 * 0.5 ** (epoch - 1)` after every fifth epoch.
    expected, rate = [], 0.001
    for epoch in range(1, 21):
        expected.append(rate)
        if epoch % 5 == 0:
            rate = 0.001 * 0.5 ** (epoch - 1)
    parameter = nn.Parameter(torch.zeros(1))
    state: dict[str, Any] = {}
    scheduler_state: dict[str, Any] = {}
    observed = []
    for _ in range(20):
        optimizer = torch.optim.Adam([parameter], lr=0.001)
        if state:
            optimizer.load_state_dict(state)
        scheduler = EpochHalving(config=EpochHalving.Config()).build(
            optimizer=optimizer, iterations=30
        )
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        state, scheduler_state = optimizer.state_dict(), scheduler.state_dict()
    assert observed == pytest.approx(expected, rel=1e-12)
    assert observed[5] == pytest.approx(0.001 * 0.5**4)


def test_gradient_norm_clip_rescales_the_update() -> None:
    def step(*, limit: float) -> float:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 1, bias=False)
                nn.init.zeros_(self.linear.weight)

            def forward(self, *, batch_x: torch.Tensor) -> torch.Tensor:
                return self.linear(batch_x)

        model = Model()
        learner = Gradient(
            config=Gradient.Config(
                learning_rate=1.0,
                max_gradient_norm=limit,
                fused_optimizer=False,
            )
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        def loss_function(*, prediction: torch.Tensor, target: torch.Tensor) -> Any:
            return (prediction - target).abs().sum() * 100

        learner.fit_epoch(
            model=model,
            optimizer=optimizer,
            batches=[(torch.ones(1, 2), torch.ones(1, 1) * 5)],
            device=torch.device("cpu"),
            loss_function=loss_function,
            regularizer=lambda: 0.0,
        )
        return float(model.linear.weight.detach().norm())

    assert step(limit=0.0) > 100
    assert step(limit=4.0) == pytest.approx(4.0, rel=1e-5)
