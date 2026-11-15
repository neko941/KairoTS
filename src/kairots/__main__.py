"""CLI overlay on the common experiment config path."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kairots.analysis.landscape import Config as LandscapeConfig
from kairots.analysis.landscape import landscape
from kairots.analysis.report import report
from kairots.config.schema import (
    _CATEGORIES,
    _DEFAULTS,
    Experiment,
    _raw_path_value,
    _required_names,
)
from kairots.registry import catalog, categories, get
from kairots.runtime.orchestrator import run


def _parse_value(*, raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.casefold() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    # A file's JSON array arrives as a list.
    if isinstance(current, (tuple, list)):
        return tuple(json.loads(raw))
    if isinstance(current, dict):
        return json.loads(raw)
    return raw


def _list(*, arguments: list[str]) -> None:
    """Print every selectable component with its declared constants."""
    parser = argparse.ArgumentParser(
        prog="python -m kairots list", description="List selectable components."
    )
    parser.add_argument("category", nargs="?", choices=categories())
    parser.add_argument("--task", help="keep components that support this task")
    parsed = parser.parse_args(arguments)
    for category, components in catalog(task=parsed.task).items():
        if parsed.category not in (None, category):
            continue
        print(category)
        for name, constants in components.items():
            details = ", ".join(
                f"{key}={value}" for key, value in sorted(constants.items())
            )
            print(f"  {name}" + (f"  ({details})" if details else ""))


def _report(*, arguments: list[str]) -> None:
    """Print a summary table of completed runs and failed run counts."""
    parser = argparse.ArgumentParser(
        prog="python -m kairots report",
        description="Report metrics across completed runs.",
    )
    parser.add_argument("root", nargs="?", default="runs", help="run directory root")
    parser.add_argument(
        "--metrics",
        default="generalization_test_pooled_mse,generalization_test_pooled_mae",
        help="comma-separated metric names to report",
    )
    parsed = parser.parse_args(arguments)
    selected_metrics = [
        item.strip() for item in parsed.metrics.split(",") if item.strip()
    ]
    print(report(root=Path(parsed.root), metrics=selected_metrics))


def _landscape(*, arguments: list[str]) -> None:
    """Compute loss surface and Hessian sharpness for a saved run."""
    parser = argparse.ArgumentParser(
        prog="python -m kairots landscape",
        description="Measure loss landscape and sharpness for a saved run.",
    )
    defaults = LandscapeConfig()
    parser.add_argument("run_dir")
    parser.add_argument("--points", type=int, default=defaults.points)
    parser.add_argument("--span", type=float, default=defaults.span)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-windows", type=int, default=defaults.max_windows)
    parser.add_argument(
        "--checkpoint", choices=("final", "best_val"), default=defaults.checkpoint
    )
    parser.add_argument(
        "--phase", choices=("train", "validation", "test"), default=defaults.phase
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--dimensions", type=int, choices=(1, 2), default=defaults.dimensions
    )
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--samples", type=int, default=defaults.samples)
    parsed = parser.parse_args(arguments)
    payload = landscape(
        run_dir=Path(parsed.run_dir),
        config=LandscapeConfig(
            points=parsed.points,
            span=parsed.span,
            seed=parsed.seed,
            max_windows=parsed.max_windows,
            checkpoint=parsed.checkpoint,
            phase=parsed.phase,
            allow_test=parsed.allow_test,
            dimensions=parsed.dimensions,
            iterations=parsed.iterations,
            samples=parsed.samples,
        ),
    )
    print(json.dumps(payload, indent=2))


def overlay(*, path: str, arguments: list[str]) -> Experiment:
    """Resolve a config file with `--namespace.field value` overrides.

    Component defaults type options without cross-component validation. All
    overrides are applied before the experiment is validated.
    """
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(arguments) % 2:
        raise ValueError("each option needs a value")
    pairs = list(zip(arguments[::2], arguments[1::2]))
    for group in (
        [pair for pair in pairs if pair[0].endswith(".name")],
        [pair for pair in pairs if not pair[0].endswith(".name")],
    ):
        for argument, raw in group:
            if not argument.startswith("--"):
                raise ValueError(f"expected --namespace.field value, got {argument}")
            parts = argument[2:].split(".")
            candidates = [
                ".".join(parts[:length])
                for length in range(1, len(parts))
                if ".".join(parts[:length]) in _CATEGORIES
            ]
            if not candidates:
                raise ValueError(
                    f"unknown configuration namespace {'.'.join(parts[:-1])}"
                )
            namespace = candidates[-1]
            namespace_parts = namespace.split(".")
            parent: dict[str, Any] = values
            for part in namespace_parts:
                parent = parent.setdefault(part, {})
                if not isinstance(parent, dict):
                    raise ValueError(
                        f"configuration namespace {namespace} must be an object"
                    )
            if parts[-1] == "name" and len(parts) == len(namespace_parts) + 1:
                if parent.get("name", _DEFAULTS[namespace]) != raw:
                    parent.clear()
                parent["name"] = raw
                continue
            selected = _required_names(values=values)
            name = _raw_path_value(values=selected, path=namespace + ".name")
            if name is None:
                name = _DEFAULTS[namespace]
            if name is None:
                raise ValueError(f"{namespace}.name must be explicit")
            component = get(
                category=_CATEGORIES[namespace],
                name=name,
            )
            defaults = asdict(component.Config())
            current: Any = {**defaults, **parent}
            for part in parts[len(namespace_parts) : -1]:
                if not isinstance(current.get(part), dict):
                    raise ValueError(
                        f"unknown configuration namespace {'.'.join(parts[:-1])}"
                    )
                current = current[part]
                parent = parent.setdefault(part, {})
            value = _parse_value(raw=raw, current=current.get(parts[-1], raw))
            parent[parts[-1]] = value
    return Experiment.from_dict(values=values)


def main() -> None:
    if sys.argv[1:2] == ["list"]:
        _list(arguments=sys.argv[2:])
        return
    if sys.argv[1:2] == ["report"]:
        _report(arguments=sys.argv[2:])
        return
    if sys.argv[1:2] == ["landscape"]:
        _landscape(arguments=sys.argv[2:])
        return
    prepare_only = sys.argv[1:2] == ["prepare"]
    if prepare_only:
        sys.argv.pop(1)
    parser = argparse.ArgumentParser(
        usage=(
            "python -m kairots [prepare|list|report|landscape] "
            "--config FILE [--namespace.field VALUE]"
        )
    )
    parser.add_argument("--config", required=True)
    known, unknown = parser.parse_known_args()
    try:
        experiment = overlay(path=known.config, arguments=unknown)
    except ValueError as error:
        parser.error(str(error))
    if prepare_only:
        store_type = get(category="data/stores", name=experiment.data.store.name)
        manifest, records = store_type(config=experiment.data.store).prepare(
            experiment=experiment
        )
        print(json.dumps({**manifest["store"], "clients": len(records)}, indent=2))
        return
    result = run(experiment=experiment)
    print(
        json.dumps(
            {
                "run_dir": result.run_dir,
                "rounds": result.rounds,
                "wall_seconds": result.wall_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
