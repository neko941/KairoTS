"""Audit every declared requirement without running the owning method."""

import copy
import inspect
import types
from dataclasses import fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints

import pytest

from kairots.config.schema import _CATEGORIES, _COMPONENT_FIELDS, Experiment
from kairots.registry import categories, get, info, names


def _set_component(*, experiment: Experiment, namespace: str, component: Any) -> None:
    parent: Any = experiment
    parts = namespace.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], component)


def _matches(*, value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        return any(_matches(value=value, annotation=argument) for argument in arguments)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_matches(value=item, annotation=arguments[0]) for item in value)
        return len(value) == len(arguments) and all(
            _matches(value=item, annotation=argument)
            for item, argument in zip(value, arguments, strict=True)
        )
    if annotation is float:
        return type(value) in (int, float)
    if annotation in (int, bool):
        return type(value) is annotation
    return isinstance(value, annotation)


def _field_type(*, experiment: Experiment, path: str) -> tuple[Any, bool]:
    parent: Any = experiment
    parts = path.split(".")
    for index, part in enumerate(parts[:-1]):
        assert hasattr(parent, part), path
        parent = getattr(parent, part)
        if parent is None:
            namespace = ".".join(parts[: index + 1])
            category = _CATEGORIES.get(namespace) or _COMPONENT_FIELDS.get(part)
            assert category is not None and parts[index + 1 :] == ["name"], path
            # An absent optional component still has a declared string name field.
            for name in names(category=category):
                config = get(category=category, name=name).Config
                assert get_type_hints(config)["name"] is str, path
            return str, True
    assert is_dataclass(parent), path
    declared = {field.name: field for field in fields(parent)}
    assert parts[-1] in declared, path
    field = declared[parts[-1]]
    return get_type_hints(type(parent))[field.name], field.default is None


def _audit(*, category: str, name: str, requires: dict[str, tuple[Any, ...]]) -> None:
    experiment = copy.deepcopy(Experiment.from_dict(values={}))
    namespace = next(
        (path for path, target in _CATEGORIES.items() if target == category), None
    )
    if namespace is not None:
        parent: Any = experiment
        for part in namespace.split("."):
            parent = getattr(parent, part)
        if parent is not None:
            _set_component(
                experiment=experiment,
                namespace=namespace,
                component=get(category=category, name=name).Config(),
            )
    # Conditional fields belong to the Config selected by singleton name rules.
    for path, allowed in requires.items():
        if path.endswith(".name") and len(allowed) == 1 and allowed[0] is not None:
            namespace = path.removesuffix(".name")
            target = _CATEGORIES.get(namespace) or _COMPONENT_FIELDS.get(
                namespace.rsplit(".", 1)[-1]
            )
            assert target is not None, (name, path)
            _set_component(
                experiment=experiment,
                namespace=namespace,
                component=get(category=target, name=allowed[0]).Config(),
            )
    for path, allowed in requires.items():
        assert isinstance(allowed, tuple) and allowed, (name, path)
        annotation, optional = _field_type(experiment=experiment, path=path)
        for value in allowed:
            if value is None:
                assert optional or type(None) in get_args(annotation), (
                    name,
                    path,
                    annotation,
                )
            else:
                assert _matches(value=value, annotation=annotation), (
                    name,
                    path,
                    value,
                    annotation,
                )


@pytest.mark.parametrize(
    "category,name",
    [
        (category, name)
        for category in categories()
        for name in names(category=category)
        if info(category=category, name=name).get("REQUIRES")
    ],
)
def test_every_requirement_path_and_value_type(*, category: str, name: str) -> None:
    _audit(
        category=category,
        name=name,
        requires=info(category=category, name=name)["REQUIRES"],
    )


@pytest.mark.parametrize("value", ["0", True, None])
def test_requirement_type_audit_rejects_invalid_values(*, value: Any) -> None:
    requires = {
        **info(category="strategies", name="SCAFFOLD")["REQUIRES"],
        "learner.optimizer.momentum": (value,),
    }
    with pytest.raises(AssertionError, match="momentum"):
        _audit(category="strategies", name="SCAFFOLD", requires=requires)


@pytest.mark.parametrize("path", ["learner.optimizer.momemtum", "training.missing"])
def test_requirement_type_audit_rejects_unknown_paths(*, path: str) -> None:
    requires = {**info(category="strategies", name="SCAFFOLD")["REQUIRES"], path: (0,)}
    with pytest.raises(AssertionError, match=path):
        _audit(category="strategies", name="SCAFFOLD", requires=requires)


def test_requirement_type_audit_checks_tuple_elements() -> None:
    requires = {
        **info(category="models", name="CSDI")["REQUIRES"],
        "task.calendar": ((7,),),
    }
    with pytest.raises(AssertionError, match="calendar"):
        _audit(category="models", name="CSDI", requires=requires)


def test_component_constructor_configs_have_dataclass_types() -> None:
    for category in categories():
        for name in names(category=category):
            component = get(category=category, name=name)
            if not hasattr(component, "Config"):
                continue
            constructor = component.__init__
            if "config" not in inspect.signature(constructor).parameters:
                continue
            owner = next(
                base for base in component.__mro__ if "__init__" in base.__dict__
            )
            annotation = get_type_hints(constructor, localns=vars(owner))["config"]
            allowed = (
                get_args(annotation)
                if get_origin(annotation) in (Union, types.UnionType)
                else (annotation,)
            )
            assert allowed and all(
                value is type(None) or is_dataclass(value) for value in allowed
            ), (category, name, annotation)


def test_flwf_client_retains_its_concrete_config_type() -> None:
    from kairots.strategies.FLwF2 import FLwF2

    assert get_type_hints(FLwF2.Client)["config"] is FLwF2.Config
