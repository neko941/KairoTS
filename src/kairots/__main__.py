"""CLI overlay on the common experiment config path."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from itertools import product
from math import prod
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


def experiments(
    *, path: str, arguments: list[str], run: str | None = None, seed: int | None = None
) -> list[tuple[str, Experiment]]:
    """Expand a config file's `sweep` and `seeds` loops into resolved experiments.

    `sweep` maps dotted fields to equal-length value lists, zipped: entry i sets
    every field to its i-th value, and is named by the first field's value. A
    list of such maps runs their product, named by the parts joined with "-". A
    file without `sweep` is one run. `run` and `seed` select one entry of each.
    """
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    sweep = values.pop("sweep", None) or {}
    seeds = values.pop("seeds", None) or [None]
    axes = []
    for group in [sweep] if isinstance(sweep, dict) and sweep else sweep:
        lengths = {len(column) for column in group.values()}
        if len(lengths) > 1:
            raise ValueError(f"{path} sweep lists differ in length: {sorted(lengths)}")
        axes.append(
            [
                (str(row[0]), dict(zip(group, row, strict=True)))
                for row in zip(*group.values(), strict=True)
            ]
        )
    runs = {
        "-".join(name for name, _ in entries): {
            field: value for _, fields in entries for field, value in fields.items()
        }
        for entries in product(*axes)
    }
    if len(runs) != prod(len(axis) for axis in axes):
        raise ValueError(f"{path} sweep names its runs twice: {sorted(runs)}")
    if run is not None:
        if run not in runs:
            raise ValueError(f"unknown run {run!r}; choose from {sorted(runs)}")
        runs = {run: runs[run]}
    if seed is not None:
        if seeds != [None] and seed not in seeds:
            raise ValueError(f"seed {seed} is not in {path}'s seeds {seeds}")
        seeds = [seed]
    selected = []
    for name, fields in runs.items():
        pairs: list[str] = []
        for field, value in fields.items():
            pairs += [
                f"--{field}",
                value if isinstance(value, str) else json.dumps(value),
            ]
        for value in seeds:
            extra = [] if value is None else ["--runtime.seed", str(value)]
            label = "/".join(
                part for part in (name, "" if value is None else f"seed{value}") if part
            )
            selected.append(
                (
                    label,
                    _overlay(
                        values=deepcopy(values), arguments=[*pairs, *extra, *arguments]
                    ),
                )
            )
    return selected


def overlay(*, path: str, arguments: list[str]) -> Experiment:
    """Resolve a single-run config file with `--namespace.field value` overrides."""
    selected = experiments(path=path, arguments=arguments)
    if len(selected) != 1:
        raise ValueError(f"{path} expands to {len(selected)} runs; select one")
    return selected[0][1]


def _overlay(*, values: dict[str, Any], arguments: list[str]) -> Experiment:
    """Apply `--namespace.field value` overrides to config values.

    Component defaults type options without cross-component validation. All
    overrides are applied before the experiment is validated.
    """
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
                # A block naming another component is reset; an unnamed block holds
                # fields shared by whichever component the sweep selects.
                if parent.get("name", raw) != raw:
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


def _reason(*, arguments: list[str]) -> None:
    """Run a text-reasoning config (its `reasoning` block, e.g. FinSTaR)."""
    from dataclasses import fields

    from kairots.reasoning import FinSTaR

    parser = argparse.ArgumentParser(
        usage="python -m kairots reason --config FILE [--reasoning.field VALUE]"
    )
    parser.add_argument("--config", required=True)
    known, unknown = parser.parse_known_args(arguments)
    values = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if "reasoning" not in values:
        parser.error("a reasoning config has a `reasoning` block")
    defaults = {item.name: item.default for item in fields(FinSTaR.Config)}
    block = values["reasoning"]
    overrides: dict[str, Any] = {}
    for flag, raw in zip(unknown[::2], unknown[1::2], strict=True):
        name = flag.removeprefix("--reasoning.")
        if name == flag:
            parser.error(f"unknown argument {flag}")
        current = block.get(name, defaults.get(name, raw))
        overrides[name] = _parse_value(raw=raw, current=current)
    result = FinSTaR.run(config=FinSTaR.config_from(values=block, overrides=overrides))
    print(json.dumps(result, indent=2))


def main() -> None:
    if sys.argv[1:2] == ["reason"]:
        _reason(arguments=sys.argv[2:])
        return
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
    parser.add_argument("--run", help="one entry of the file's runs")
    parser.add_argument("--seed", type=int, help="one entry of the file's seeds")
    known, unknown = parser.parse_known_args()
    try:
        selected = experiments(
            path=known.config, arguments=unknown, run=known.run, seed=known.seed
        )
    except ValueError as error:
        parser.error(str(error))
    outputs = []
    for label, experiment in selected:
        if prepare_only:
            store_type = get(category="data/stores", name=experiment.data.store.name)
            manifest, records = store_type(config=experiment.data.store).prepare(
                experiment=experiment
            )
            output = {**manifest["store"], "clients": len(records)}
        else:
            result = run(experiment=experiment)
            output = {
                "run_dir": result.run_dir,
                "rounds": result.rounds,
                "wall_seconds": result.wall_seconds,
            }
        outputs.append({"run": label, **output} if label else output)
    print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, indent=2))


if __name__ == "__main__":
    main()
