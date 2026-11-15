"""Resolve the training run used by zero-shot evaluation."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from kairots.config.schema import Experiment
from kairots.registry import get, info


def resolve_source(*, experiment: Experiment) -> Experiment:
    """Reuse a completed compatible source run, or train the source config once."""
    if experiment.initialization.run_dir:
        return experiment
    if experiment.runtime.resume:
        saved = json.loads(
            (Path(experiment.runtime.resume) / "config.json").read_text(
                encoding="utf-8"
            )
        )["resolved"]
        return replace(
            experiment,
            initialization=replace(
                experiment.initialization, run_dir=saved["initialization"]["run_dir"]
            ),
        )
    path = Path(experiment.strategy.source_config)
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("ZeroShot source_config must contain an experiment object")
    source = Experiment.from_dict(values=values)
    if (
        info(category="strategies", name=source.strategy.name).get("SOURCE_RUN", False)
        or source.runtime.resume
        or source.groups
    ):
        raise ValueError(
            "ZeroShot source config needs an independent ungrouped training run"
        )
    if source.model.name != experiment.model.name or any(
        value != asdict(experiment.model).get(key)
        for key, value in asdict(source.model).items()
        if key != "input_channels"
    ):
        raise ValueError("ZeroShot requires matching source and target model configs")
    source = replace(source, runtime=replace(source.runtime, save_checkpoints=True))
    store_type = get(category="data/stores", name=source.data.store.name)
    manifest, _ = store_type(config=source.data.store).prepare(experiment=source)
    expected = json.loads(json.dumps(source.to_dict(), default=str))
    group = Path(source.runtime.runs_dir) / (
        f"{source.data.source.name}-{source.strategy.name}-{source.model.name}"
    )
    model_type = get(category="models", name=source.model.name)
    shapes = {
        name: value.shape
        for name, value in model_type(config=source.model).state_dict().items()
    }
    checkpoint = experiment.initialization.checkpoint
    selected: Path | None = None
    for candidate in sorted(group.iterdir(), reverse=True) if group.is_dir() else []:
        if not candidate.is_dir():
            continue
        try:
            summary = _read(directory=candidate, name="summary")
            config = _read(directory=candidate, name="config")
            prepared = _read(directory=candidate, name="manifest")
            if (
                summary.get("status") != "completed"
                or config.get("resolved") != expected
                or prepared.get("store") != manifest["store"]
            ):
                continue
            state = torch.load(
                candidate / "checkpoints" / f"{checkpoint}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if (
                not isinstance(state, dict)
                or set(state) != set(shapes)
                or any(
                    not isinstance(value, torch.Tensor) or value.shape != shapes[name]
                    for name, value in state.items()
                )
            ):
                continue
        except OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError:
            continue
        selected = candidate
        break
    if selected is None:
        from kairots.runtime.orchestrator import run

        selected = Path(run(experiment=source).run_dir)
        if not (selected / "checkpoints" / f"{checkpoint}.pt").is_file():
            raise ValueError(
                f"source training did not produce checkpoint {checkpoint!r}"
            )
    return replace(
        experiment,
        initialization=replace(experiment.initialization, run_dir=str(selected)),
    )


def _read(*, directory: Path, name: str) -> dict[str, Any]:
    value = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{directory / name} must contain a JSON object")
    return value
