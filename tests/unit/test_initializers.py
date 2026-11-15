from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kairots.config.schema import Experiment
from kairots.initializers.Average import Average
from kairots.initializers.Rules import Rules
from kairots.models.DLinear import DLinear

CLASSIFIER_RULES = (
    ("body.linear_seasonal.*.weight", "inverse_fan_in", {}),
    ("body.linear_trend.*.weight", "inverse_fan_in", {}),
    ("head.projection.weight", "xavier_uniform", {}),
    ("head.projection.bias", "zeros", {}),
)


def test_average_supports_other_models_and_preserves_non_linear_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    native = {name: value.clone() for name, value in model.named_parameters()}
    policy = Average(config=Average.Config())
    assert policy.apply(model=model) == {"0.weight": "inverse_fan_in"}
    torch.testing.assert_close(model[0].weight, torch.full((2, 3), 1 / 3))
    for name, value in model.named_parameters():
        if name != "0.weight":
            torch.testing.assert_close(value, native[name])
    root = torch.nn.Linear(4, 2)
    assert policy.apply(model=root) == {"weight": "inverse_fan_in"}
    torch.testing.assert_close(root.weight, torch.full((2, 4), 1 / 4))


def test_average_patterns_validate_and_can_reproduce_dlinear_preset() -> None:
    model = _classifier()
    policy = Average(config=Average.Config(patterns=("body.*",)))
    assert set(policy.apply(model=model)) == {
        "body.linear_seasonal.0.weight",
        "body.linear_trend.0.weight",
    }
    for patterns in (("missing.*",), (), ("*", "*weight")):
        with pytest.raises(ValueError):
            Average(config=Average.Config(patterns=patterns)).apply(model=model)
    Experiment.from_dict(
        values={"model": {"name": "Linear"}, "initialization": {"name": "Average"}}
    )


def _classifier() -> DLinear:
    torch.manual_seed(0)
    return DLinear(
        config=DLinear.Config(
            head="Classification", num_classes=4, input_length=10, input_channels=3
        )
    )


def test_policy_sets_only_its_rules_and_keeps_native_values() -> None:
    native, model = _classifier(), _classifier()
    applied = Rules(config=Rules.Config(rules=CLASSIFIER_RULES)).apply(model=model)
    assert applied == {
        "body.linear_seasonal.0.weight": "inverse_fan_in",
        "body.linear_trend.0.weight": "inverse_fan_in",
        "head.projection.weight": "xavier_uniform",
        "head.projection.bias": "zeros",
    }
    parameters = dict(model.named_parameters())
    native_parameters = dict(native.named_parameters())
    assert torch.equal(parameters["head.projection.bias"], torch.zeros(4))
    assert not torch.equal(
        parameters["head.projection.weight"],
        native_parameters["head.projection.weight"],
    )
    for name in ("body.linear_seasonal.0.weight", "body.linear_trend.0.weight"):
        assert torch.equal(parameters[name], torch.full((10, 10), 1 / 10))
    for name in ("body.linear_seasonal.0.bias", "body.linear_trend.0.bias"):
        assert torch.equal(parameters[name], native_parameters[name])


def test_policy_rules_must_match_once() -> None:
    forecaster = DLinear(config=DLinear.Config())
    with pytest.raises(ValueError, match="matches no parameter"):
        Rules(config=Rules.Config(rules=CLASSIFIER_RULES)).apply(model=forecaster)

    with pytest.raises(ValueError, match="two rules"):
        Rules(
            config=Rules.Config(
                rules=(("head.*", "zeros", {}), ("head.projection.bias", "ones", {}))
            )
        ).apply(model=_classifier())


def test_policy_initializes_a_tied_parameter_once_and_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kairots.initializers.base import INITIALIZERS

    class Tied(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = torch.nn.Linear(2, 2, bias=False)
            self.right = torch.nn.Linear(2, 2, bias=False)
            self.right.weight = self.left.weight

    calls = 0

    def counted(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.nn.init.zeros_(tensor)

    monkeypatch.setitem(INITIALIZERS, "counted", counted)
    model = Tied()
    policy = Rules(config=Rules.Config(rules=(("*.weight", "counted", {}),)))
    assert policy.apply(model=model) == {"left.weight": "counted"}
    assert calls == 1
    assert model.left.weight is model.right.weight
    assert torch.equal(model.left.weight, torch.zeros_like(model.left.weight))

    conflict = Rules(
        config=Rules.Config(
            rules=(("left.weight", "zeros", {}), ("*.weight", "ones", {}))
        )
    )
    with pytest.raises(ValueError, match="two rules"):
        conflict.apply(model=Tied())


def test_pretrained_keeps_a_new_head_at_its_native_initialization(
    workspace_tmp_path: Path,
) -> None:
    from kairots.initializers.Pretrained import Pretrained

    class Target(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.head = torch.nn.Linear(2, 1, bias=False)

    source = workspace_tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    torch.save(
        {"encoder.weight": torch.full((2, 2), 7.0)},
        source / "checkpoints" / "final.pt",
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "clients": [
                    {
                        "name": "client",
                        "cut_timestamps": {"validation": None, "test": 10},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    model = Target()
    with torch.no_grad():
        model.head.weight.fill_(42.0)
    head = model.head.weight.clone()
    manifest = {
        "clients": [
            {
                "name": "client",
                "cut_timestamps": {"validation": None, "test": 20},
            }
        ]
    }

    loaded = Pretrained(config=Pretrained.Config(run_dir=str(source))).apply(
        model=model, manifest=manifest
    )

    assert loaded == {"encoder.weight": "pretrained"}
    assert torch.equal(model.encoder.weight, torch.full((2, 2), 7.0))
    assert torch.equal(model.head.weight, head)


def test_pretrained_evaluation_checks_panel_sample_disjointness(
    workspace_tmp_path: Path,
) -> None:
    from kairots.initializers.Pretrained import Pretrained

    source = workspace_tmp_path / "panel-source"
    (source / "checkpoints").mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    torch.save(model.state_dict(), source / "checkpoints" / "final.pt")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "data_permissions": {"mode": "fit"},
                "dataset_family": "Panel",
                "clients": [
                    {
                        "name": "patients",
                        "windows": {
                            "train": [[0, 2799]],
                            "validation": [[2800, 3199]],
                            "test": [[3200, 3999]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = Pretrained(config=Pretrained.Config(run_dir=str(source)))
    target = {
        "data_permissions": {"mode": "evaluate"},
        "dataset_family": "Panel",
        "clients": [{"name": "patients", "windows": {"test": [[3200, 3999]]}}],
    }

    loaded = policy.apply(model=model, manifest=target)

    assert loaded
    assert target["pretraining_calendar_check"]["cross_population"] == (
        "sample-disjoint"
    )
    target["clients"][0]["windows"]["test"] = [[100, 110]]
    with pytest.raises(ValueError, match="samples overlap evaluation samples"):
        policy.apply(model=model, manifest=target)


def test_rules_validate_initializer_names_and_keep_instance_rules() -> None:
    with pytest.raises(ValueError, match="unknown initializer"):
        Experiment.from_dict(
            values={
                "initialization": {"name": "Rules", "rules": [["*", "missing", {}]]}
            }
        )
    first = Rules(config=Rules.Config(rules=(("weight", "zeros", {}),)))
    second = Rules(config=Rules.Config(rules=(("weight", "ones", {}),)))
    assert first.RULES != second.RULES
    model = torch.nn.Linear(2, 2)
    first.apply(model=model)
    assert torch.equal(model.weight, torch.zeros_like(model.weight))
    second.apply(model=model)
    assert torch.equal(model.weight, torch.ones_like(model.weight))


def test_configured_rules_reproduce_the_four_parameter_initializers() -> None:
    expected = _classifier()
    parameters = dict(expected.named_parameters())
    with torch.no_grad():
        for name in ("body.linear_seasonal.0.weight", "body.linear_trend.0.weight"):
            torch.nn.init.constant_(parameters[name], 0.1)
        torch.nn.init.xavier_uniform_(parameters["head.projection.weight"])
        torch.nn.init.zeros_(parameters["head.projection.bias"])
    actual = _classifier()
    values = {"initialization": {"name": "Rules", "rules": CLASSIFIER_RULES}}
    config = Experiment.from_dict(values=values).initialization
    Rules(config=config).apply(model=actual)
    for name, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name])


def test_pretrained_copies_checkpoint_and_refuses_held_out_overlap(
    workspace_tmp_path: Path,
) -> None:
    from fixtures.synthetic import ett_experiment, write_ett_sources

    from kairots.initializers.Pretrained import Pretrained
    from kairots.registry import get
    from kairots.runtime.orchestrator import run

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    pretrain = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "pretrain", save_checkpoints=True
    )
    source_dir = Path(run(experiment=pretrain).run_dir)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(source_dir / "checkpoints" / "final.pt", weights_only=True)
    model = get(category="models", name=pretrain.model.name)(config=pretrain.model)
    # A run group resolves to its latest run holding the checkpoint, so a newer
    # run without one (the run being initialized) is skipped.
    (source_dir.parent / "zz-new-run").mkdir()
    policy = Pretrained(config=Pretrained.Config(run_dir=str(source_dir.parent)))
    applied = policy.apply(model=model, manifest=manifest)
    assert sorted(applied) == sorted(state)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name])
    # Fine-tuning held out from an earlier timestamp than pretraining did: refused.
    earlier = json.loads(json.dumps(manifest))
    for client in earlier["clients"]:
        cut = client["cut_timestamps"]
        key = "validation" if cut["validation"] is not None else "test"
        cut[key] = cut[key] - 1
    with pytest.raises(ValueError, match="past this run's first held-out"):
        policy.apply(model=model, manifest=earlier)
    # Another population (only ETTh1 here) is held to this run's earliest cut.
    target = json.loads(json.dumps(manifest["clients"][:1]))
    target[0]["name"] = "Other/Other"
    policy.apply(model=model, manifest={"clients": target})
    cut = target[0]["cut_timestamps"]
    cut["validation" if cut["validation"] is not None else "test"] -= 1
    with pytest.raises(ValueError, match="past this run's first held-out"):
        policy.apply(model=model, manifest={"clients": target})
    # Validation selected a best_val checkpoint, so it saw up to the test cut.
    with pytest.raises(ValueError, match="past this run's first held-out"):
        Pretrained(
            config=Pretrained.Config(run_dir=str(source_dir), checkpoint="best_val")
        ).apply(model=model, manifest=manifest)
    # End to end: the orchestrator hands the policy this run's manifest.
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "finetune"
    ).to_dict()
    values["initialization"] = {"name": "Pretrained", "run_dir": str(source_dir)}
    fine_dir = Path(run(experiment=Experiment.from_dict(values=values)).run_dir)
    fine = json.loads((fine_dir / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(fine["initialization"]["parameters"]) == sorted(state)
