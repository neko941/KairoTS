"""AST-only lazy component discovery."""

from __future__ import annotations

import ast
import functools
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

PACKAGE_ROOT = Path(__file__).parent


def _constants(*, node: ast.ClassDef) -> dict[str, Any]:
    """Read a class's literal UPPER_CASE constants (TASKS, NEEDS, ...) via AST."""
    found: dict[str, Any] = {}
    for statement in node.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                try:
                    found[target.id] = ast.literal_eval(statement.value)
                except ValueError:
                    continue
    return found


@functools.cache
def _components(*, category: str) -> dict[str, tuple[str, dict[str, Any]]]:
    directory = PACKAGE_ROOT / category
    if not directory.is_dir():
        raise KeyError(f"unknown component category {category!r}")
    found: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted(directory.rglob("*.py")):
        if path.name.startswith("_") and path.name != "__init__.py":
            continue
        if path.stem in {"base", "contracts"}:
            continue
        # A model is its directory's recipe; heads and backbones inside it are parts.
        if category == "models" and path.name != "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_parts = path.relative_to(PACKAGE_ROOT).with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
            component_name = module_parts[-1]
        else:
            component_name = path.stem
        selected = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == component_name
            ),
            None,
        )
        if selected is None:
            continue
        constants = _constants(node=selected)
        if constants.get("ABSTRACT") is True:
            continue
        if component_name in found:
            raise ValueError(
                f"{category} component {component_name!r} is defined twice: "
                f"{found[component_name][0]} and {'.'.join(module_parts)}"
            )
        found[component_name] = (".".join(module_parts), constants)
    return found


def names(*, category: str, task: str | None = None) -> tuple[str, ...]:
    """List selectable components without importing them; `task` keeps compatible ones.

    A component without a `TASKS` constant works with every task.
    """
    return tuple(
        name
        for name, (_, constants) in sorted(_components(category=category).items())
        if task is None or task in constants.get("TASKS", (task,))
    )


def info(*, category: str, name: str) -> dict[str, Any]:
    """Return a component's declared constants (TASKS, NEEDS, DIRECTION, ...)."""
    if name not in _components(category=category):
        raise KeyError(f"unknown {category} component {name!r}")
    return _components(category=category)[name][1]


def module(*, category: str, name: str) -> str:
    """Return the dotted module that defines a component, without importing it."""
    if name not in _components(category=category):
        raise KeyError(f"unknown {category} component {name!r}")
    return f"kairots.{_components(category=category)[name][0]}"


def require(
    *,
    category: str,
    name: str,
    task: str,
    phase: str | None = None,
    role: str | None = None,
) -> None:
    """Fail before a run when a component's TASKS or PHASES exclude this use."""
    constants = info(category=category, name=name)
    if role is not None and role not in constants.get("ROLES", ()):
        raise ValueError(f"{category} {name!r} does not support the {role!r} role")
    tasks = constants.get("TASKS")
    if tasks is not None and task not in tasks:
        raise ValueError(
            f"{category} {name!r} does not support task {task!r}; it supports {tasks}"
        )
    phases = constants.get("PHASES")
    if phase is not None and phases is not None and phase not in phases:
        raise ValueError(
            f"{category} {name!r} cannot be selected for the {phase} phase; "
            f"it allows {phases}"
        )


def categories() -> tuple[str, ...]:
    """List leaf packages whose __init__ exposes a registry surface (`data/sources`).

    Parent surfaces such as `data` would repeat their children, so they are left out.
    """
    found = {
        path.parent.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("__init__.py")
        if "surface(package=__name__)" in path.read_text(encoding="utf-8")
    }
    return tuple(
        sorted(
            category
            for category in found
            if not any(other.startswith(category + "/") for other in found)
        )
    )


def get(*, category: str, name: str) -> Any:
    """Import one selectable implementation by name."""
    # Components are duck-typed by category, so callers get Any, not type[object].
    if name not in _components(category=category):
        raise KeyError(f"unknown {category} component {name!r}")
    module_path = _components(category=category)[name][0]
    module: ModuleType = importlib.import_module(f"kairots.{module_path}")
    component = getattr(module, name)
    if not isinstance(component, type):
        raise TypeError(f"{module.__name__}.{name} is not a class")
    # Importing kairots.<category>.<Name> binds the submodule on the category package;
    # rebind the class so `from kairots.<category> import <Name>` returns the class.
    category_package = importlib.import_module("kairots." + category.replace("/", "."))
    setattr(category_package, name, component)
    return component


def surface(*, package: str) -> tuple[Any, Any, Any, Any]:
    """Build a category package's lazy `get`, `names`, `__getattr__` and `__dir__`."""
    category = package.removeprefix("kairots.").replace(".", "/")

    def category_get(*, name: str) -> Any:
        return get(category=category, name=name)

    def category_names(*, task: str | None = None) -> tuple[str, ...]:
        return names(category=category, task=task)

    def module_getattr(name: str) -> Any:  # positional: PEP 562 hook.
        if name in category_names():
            return category_get(name=name)
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    def module_dir() -> list[str]:
        return sorted({"get", "names", *category_names()})

    return category_get, category_names, module_getattr, module_dir


def catalog(*, task: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """Every category's selectable components and their declared constants."""
    return {
        category: {
            name: info(category=category, name=name)
            for name in names(category=category, task=task)
        }
        for category in categories()
    }
