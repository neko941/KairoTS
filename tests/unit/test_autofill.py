"""Only omitted fields may inherit a component's singleton requirement."""

import json
from pathlib import Path
from typing import Any

import pytest

from kairots.__main__ import overlay
from kairots.config.schema import Experiment


def test_unset_requirement_is_filled_without_mutating_input() -> None:
    values = {"strategy": {"name": "SCAFFOLD"}}
    experiment = Experiment.from_dict(values=values)
    assert experiment.learner.optimizer.name == "SGD"
    assert values == {"strategy": {"name": "SCAFFOLD"}}


@pytest.mark.parametrize("optimizer", ["Adam", "SGD"])
def test_explicit_requirement_even_when_equal_to_default(*, optimizer: str) -> None:
    values = {
        "strategy": {"name": "SCAFFOLD"},
        "learner": {"optimizer": {"name": optimizer}},
    }
    if optimizer == "Adam":
        with pytest.raises(ValueError, match="SCAFFOLD.*learner.optimizer.name.*SGD"):
            Experiment.from_dict(values=values)
    else:
        assert Experiment.from_dict(values=values).learner.optimizer.name == "SGD"


def test_cli_requirement_override_is_explicit(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"strategy": {"name": "SCAFFOLD"}}))
    assert overlay(path=str(config), arguments=[]).learner.optimizer.name == "SGD"
    with pytest.raises(ValueError, match="SCAFFOLD.*learner.optimizer.name"):
        overlay(path=str(config), arguments=["--learner.optimizer.name", "Adam"])
    assert (
        overlay(
            path=str(config), arguments=["--learner.optimizer.name", "SGD"]
        ).learner.optimizer.name
        == "SGD"
    )


def test_cli_types_options_using_the_required_component(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"strategy": {"name": "SCAFFOLD"}}))
    experiment = overlay(
        path=str(config), arguments=["--learner.optimizer.momentum", "0"]
    )
    assert experiment.learner.optimizer.name == "SGD"
    assert experiment.learner.optimizer.momentum == 0.0
    config.write_text(json.dumps({"strategy": {"name": "ZeroShot"}}))
    experiment = overlay(
        path=str(config), arguments=["--initialization.run_dir", str(tmp_path)]
    )
    assert experiment.initialization.name == "Pretrained"
    assert experiment.initialization.run_dir == str(tmp_path)


def test_required_component_names_resolve_before_their_fields() -> None:
    experiment = Experiment.from_dict(values={"strategy": {"name": "ZeroShot"}})
    assert experiment.model.name == "Linear"
    assert experiment.initialization.name == "Pretrained"


def test_required_names_precede_config_specific_options() -> None:
    experiment = Experiment.from_dict(
        values={
            "strategy": {"name": "SCAFFOLD"},
            "learner": {"optimizer": {"momentum": 0.0}},
        }
    )
    assert experiment.learner.optimizer.name == "SGD"
    experiment = Experiment.from_dict(
        values={
            "strategy": {"name": "ZeroShot"},
            "model": {"name": "DLinear", "individual": False},
        }
    )
    assert experiment.model.name == "DLinear"


def test_multiple_allowed_values_never_pick_a_default() -> None:
    with pytest.raises(ValueError, match="Denoising.*model.name"):
        Experiment.from_dict(values={"learner": {"name": "Denoising"}})


def test_explicit_absent_namespace_is_not_replaced() -> None:
    with pytest.raises(ValueError, match="FLwF2.*training.name"):
        Experiment.from_dict(
            values={
                "task": {"name": "Classification"},
                "data": {
                    "source": {"name": "BasicMotions"},
                    "split": {"name": "Predefined"},
                },
                "strategy": {"name": "FLwF2"},
                "model": {"name": "DLinear"},
                "training": None,
            }
        )


def test_newly_selected_component_requirements_are_filled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kairots.config.schema as schema

    original = schema.info

    def constrained(*, category: str, name: str) -> dict[str, Any]:
        constants = original(category=category, name=name)
        if (category, name) == ("models", "DLinear"):
            return {**constants, "REQUIRES": {"runtime.seed": (17,)}}
        return constants

    monkeypatch.setattr(schema, "info", constrained)
    experiment = Experiment.from_dict(
        values={"strategy": {"name": "ZeroShot"}, "model": {"name": "DLinear"}}
    )
    assert experiment.model.name == "DLinear"
    assert experiment.runtime.seed == 17


def test_conflicting_singleton_requirements_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kairots.config.schema as schema

    original = schema.info

    def constrained(*, category: str, name: str) -> dict[str, Any]:
        constants = original(category=category, name=name)
        if (category, name) in {("models", "Linear"), ("optimizers", "Adam")}:
            seed = 17 if category == "models" else 19
            return {**constants, "REQUIRES": {"runtime.seed": (seed,)}}
        return constants

    monkeypatch.setattr(schema, "info", constrained)
    with pytest.raises(ValueError, match="conflicting.*runtime.seed"):
        Experiment.from_dict(values={})


def test_groups_keep_their_own_omitted_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    import kairots.config.schema as schema

    original = schema.info

    def constrained(*, category: str, name: str) -> dict[str, Any]:
        constants = original(category=category, name=name)
        if category == "data/sources" and name in {"ETDatasetHour", "ETTh1"}:
            seed = 17 if name == "ETDatasetHour" else 19
            return {**constants, "REQUIRES": {"runtime.seed": (seed,)}}
        return constants

    monkeypatch.setattr(schema, "info", constrained)
    experiment = Experiment.from_dict(
        values={"groups": [{"data": {"source": {"name": "ETTh1"}}}]}
    )
    assert experiment.runtime.seed == 17
    assert experiment.groups[0].runtime.seed == 19
