"""Every component-owned compatibility rule rejects an incompatible experiment."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from kairots.config.schema import (
    _CATEGORIES,
    _COMPONENT_FIELDS,
    Experiment,
    _selected_components,
)
from kairots.registry import categories, info, names

CLASSIFICATION = {
    "task": {"name": "Classification"},
    "data": {"source": {"name": "BasicMotions"}, "split": {"name": "Predefined"}},
    "model": {"name": "DLinear"},
    "training": {"name": "ClassIncremental", "periods": 2},
}
BASES = {
    "csdi": {
        "task": {"name": "ProbabilisticImputation"},
        "model": {"name": "CSDI"},
        "learner": {"name": "Denoising"},
    },
    "timegrad": json.loads(
        (
            Path(__file__).parents[2]
            / "configs/sample_forecasting/ExchangeRate-TimeGrad.json"
        ).read_text()
    ),
    "denoising": {
        "task": {"name": "ProbabilisticImputation"},
        "model": {"name": "CSDI"},
        "learner": {"name": "Denoising"},
    },
    "continual": CLASSIFICATION,
    "flwf2": {
        **CLASSIFICATION,
        "strategy": {"name": "FLwF2"},
        "learner": {"optimizer": {"name": "SGD"}},
    },
    "zero": {
        "strategy": {"name": "ZeroShot"},
        "model": {"name": "DLinear"},
        "initialization": {"name": "Pretrained", "run_dir": "runs/source"},
    },
    "scaffold": {
        "strategy": {"name": "SCAFFOLD"},
        "learner": {"optimizer": {"name": "SGD"}},
    },
    "prequential": {"evaluation": {"name": "Prequential"}},
}


@pytest.mark.parametrize(
    "base,namespace,path,value",
    [
        ("csdi", "model", "task.calendar", ["DayOfWeek"]),
        ("csdi", "model", "learner.name", "Gradient"),
        ("csdi", "model", "learner.precision", "bf16"),
        ("timegrad", "model", "learner.name", "Gradient"),
        ("timegrad", "model", "learner.precision", "bf16"),
        ("denoising", "learner", "model.name", "Naive"),
        ("flwf2", "strategy", "training.name", "Online"),
        ("flwf2", "strategy", "learner.name", "REINFORCE"),
        ("flwf2", "strategy", "learner.optimizer.name", "Adam"),
        ("flwf2", "strategy", "learner.precision", "bf16"),
        ("continual", "training", "task.name", "Forecasting"),
        ("continual", "training", "groups", [{}]),
        ("continual", "training", "strategy.name", "LocalOnly"),
        ("continual", "training", "evaluation.name", "Adapted"),
        ("continual", "training", "evaluation.finetune_epochs", 1),
        ("zero", "strategy", "groups", [{}]),
        ("zero", "strategy", "client_model", {"name": "DLinear"}),
        ("zero", "strategy", "initialization.name", "Random"),
        ("zero", "strategy", "evaluation.finetune_epochs", 1),
        ("scaffold", "strategy", "learner.optimizer.name", "Adam"),
        ("scaffold", "strategy", "learner.optimizer.momentum", 0.9),
        ("scaffold", "strategy", "learner.optimizer.weight_decay", 0.01),
        ("prequential", "evaluation", "evaluation.regime", "Continual"),
    ],
)
def test_each_declared_requirement_rejects_a_violation(
    *,
    base: str,
    namespace: str,
    path: str,
    value: Any,
) -> None:
    valid = Experiment.from_dict(values=BASES[base])
    component = getattr(valid, namespace)
    assert path in info(category=_CATEGORIES[namespace], name=component.name).get(
        "REQUIRES", {}
    )
    values = deepcopy(BASES[base])
    parent = values
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent.setdefault(part, {})
    if parts[-1] == "name":
        parent.clear()
    parent[parts[-1]] = value
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        Experiment.from_dict(values=values)


def test_requirement_paths_are_strict_and_optional_namespaces_are_none() -> None:
    from kairots.config.schema import _path_value

    experiment = Experiment.from_dict(values={})
    assert _path_value(experiment=experiment, path="training.name") is None
    assert _path_value(experiment=experiment, path="model.name") == "Linear"
    with pytest.raises(AttributeError):
        _path_value(experiment=experiment, path="model.missing")
    for namespace in ("model", "learner", "strategy", "evaluation", "initialization"):
        component = getattr(experiment, namespace)
        for path in info(category=_CATEGORIES[namespace], name=component.name).get(
            "REQUIRES", {}
        ):
            _path_value(experiment=experiment, path=path)


def test_training_needs_an_explicit_name_and_rejects_online_at_top_level() -> None:
    with pytest.raises(ValueError, match=r"training\.name"):
        Experiment.from_dict(values={"training": {}})
    with pytest.raises(ValueError, match="training"):
        Experiment.from_dict(values={"training": {"name": "Online"}})
    with pytest.raises(KeyError, match="Missing"):
        Experiment.from_dict(values={"training": {"name": "Missing"}})


def test_continual_period_limit_is_checked_by_its_component() -> None:
    values = deepcopy(CLASSIFICATION)
    values["training"]["periods"] = 5
    with pytest.raises(ValueError, match="num_classes"):
        Experiment.from_dict(values=values)


@pytest.mark.parametrize(
    "namespace,category,name,values",
    [
        ("learner.optimizer", "optimizers", "Adam", {}),
        ("data.partition", "data/partitions", "PerFile", {}),
        (
            "client_model",
            "models",
            "TimesNet",
            {
                "task": {"name": "Classification"},
                "data": CLASSIFICATION["data"],
                "model": {"name": "DLinear"},
                "client_model": {"name": "TimesNet"},
                "strategy": {"name": "FML"},
                "learner": {"optimizer": {"name": "SGD"}},
            },
        ),
        (
            "evaluation.regime",
            "training",
            "Online",
            {"evaluation": {"name": "Prequential"}},
        ),
        (
            "evaluation.replay",
            "replay",
            "Reservoir",
            {"evaluation": {"name": "Prequential"}},
        ),
        (
            "evaluation.adaptation",
            "adaptation",
            "TENT",
            {
                "task": {"name": "Classification"},
                "data": CLASSIFICATION["data"],
                "model": {"name": "DLinear"},
                "evaluation": {"name": "Adapted"},
            },
        ),
        (
            "strategy.topology",
            "topologies",
            "FullyConnected",
            {"strategy": {"name": "DFedAvg"}},
        ),
    ],
)
def test_nested_component_requirements_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
    category: str,
    name: str,
    values: dict[str, Any],
) -> None:
    import kairots.config.schema as schema

    original = schema.info

    def constrained(*, category: str, name: str) -> dict[str, Any]:
        constants = original(category=category, name=name)
        if (category, name) == target:
            return {**constants, "REQUIRES": {"runtime.seed": (-1,)}}
        return constants

    target = category, name
    monkeypatch.setattr(schema, "info", constrained)
    with pytest.raises(
        ValueError, match=namespace.replace(".", r"\.") + ".*runtime.seed"
    ):
        Experiment.from_dict(values={**values, "runtime": {"seed": 941}})


def test_declared_constraints_have_a_walked_category() -> None:
    reachable = set(_COMPONENT_FIELDS.values()) | set(_CATEGORIES.values())
    for category in categories():
        for name in names(category=category):
            constants = info(category=category, name=name)
            if "REQUIRES" in constants or "TASKS" in constants:
                assert category in reachable, (category, name)
    experiment = Experiment.from_dict(values={"evaluation": {"name": "Prequential"}})
    assert ("evaluation.regime", "training", "Online") in set(
        _selected_components(experiment=experiment)
    )


def test_nested_component_tasks_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    import kairots.config.schema as schema
    import kairots.registry as registry

    original = registry.info

    def restricted(*, category: str, name: str) -> dict[str, Any]:
        constants = original(category=category, name=name)
        if (category, name) == ("optimizers", "Adam"):
            return {**constants, "TASKS": ("Classification",)}
        return constants

    monkeypatch.setattr(registry, "info", restricted)
    monkeypatch.setattr(schema, "info", restricted)
    with pytest.raises(ValueError, match="optimizers.*does not support task"):
        Experiment.from_dict(values={})


def test_nested_component_validate_is_called(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairots.registry import get

    def validate(*, experiment: Experiment) -> None:
        raise ValueError("nested optimizer validation")

    monkeypatch.setattr(
        get(category="optimizers", name="Adam"),
        "validate",
        staticmethod(validate),
        raising=False,
    )
    with pytest.raises(ValueError, match="nested optimizer validation"):
        Experiment.from_dict(values={})
