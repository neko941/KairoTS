"""Resolve all experiment namespaces through the lazy component registry."""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import InitVar, asdict, dataclass, fields, is_dataclass
from functools import partial
from typing import Any

from kairots.data.sources.Irregular import Irregular
from kairots.registry import get, info, require


@dataclass(kw_only=True)
class DataConfig:
    source: Any
    partition: Any
    split: Any
    scaler: Any
    view: Any
    store: Any
    # Optional: training-input augmentation, off unless configured.
    augmentation: Any = None


@dataclass(kw_only=True)
class Reproduces:
    """The published setting a config reproduces exactly."""

    # Versioned paper identifier, such as "arXiv:2205.13504v3".
    paper: str
    # Table or figure holding the reproduced numbers, such as "Table 2".
    table: str
    # Cell label to the paper's printed value, such as {"ETTh1 96 MSE": 0.375}; a
    # horizon or dataset sweep lists one cell per run.
    cells: dict[str, float]
    # Official code pinned to a commit; empty when the paper releases none.
    code: str = ""

    def __post_init__(self) -> None:
        if not self.paper or not self.table or not self.cells:
            raise ValueError("reproduces needs a paper, a table and at least one cell")
        for label, value in self.cells.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"reproduces cell {label!r} needs the paper's number")


@dataclass(kw_only=True)
class Experiment:
    task: Any
    data: DataConfig
    model: Any
    learner: Any
    scheduler: Any
    initialization: Any
    strategy: Any
    evaluation: Any
    runtime: Any
    # The model each client trains and is scored with, when a model-heterogeneous
    # strategy (one declaring CLIENT_MODEL) shares another one (`model`); None:
    # clients own the shared model.
    training: Any = None
    client_model: Any = None
    # Client groups, each with its own data and client model over this experiment's
    # other settings; clients are numbered in group
    # order. Empty: every client belongs to this experiment.
    groups: tuple[Experiment, ...] = ()
    # Partition client ids a group keeps;
    # empty keeps every client.
    clients: tuple[int, ...] = ()
    # The paper setting this config reproduces; None for an example config.
    reproduces: Reproduces | None = None
    _validate: InitVar[bool] = True

    @classmethod
    def from_dict(cls, *, values: dict[str, Any]) -> Experiment:
        """Fill singleton requirements only where the file/CLI omitted a value."""
        provided, leaves = _provided_paths(values=values)
        pending = deepcopy(values)
        seen: set[str] = set()
        updates: dict[str, Any] = {}
        experiment: Experiment | None = None
        while True:
            state = json.dumps(pending, sort_keys=True, default=str)
            if state in seen:
                raise ValueError(
                    f"conflicting component REQUIRES for {', '.join(updates)}"
                )
            seen.add(state)
            updates = {}
            # Required names must precede Config construction, then field validation.
            for raw_names in (True, False):
                if raw_names:
                    selected = _selected_components(experiment=pending)
                    read_value = partial(_raw_path_value, values=pending)
                else:
                    experiment = cls._from_dict(values=pending, original=values)
                    selected = tuple(_selected_components(experiment=experiment))
                    read_value = partial(_path_value, experiment=experiment)
                selected = tuple(selected)
                for name_fields in (True,) if raw_names else (True, False):
                    for namespace, category, name in selected:
                        for path, allowed in (
                            info(category=category, name=name)
                            .get("REQUIRES", {})
                            .items()
                        ):
                            explicit = path in provided or any(
                                path.startswith(key + ".") for key in leaves
                            )
                            if path.endswith(".name") != name_fields and not explicit:
                                continue
                            value = read_value(path=path)
                            if value in allowed:
                                continue
                            if explicit:
                                raise ValueError(
                                    f"{namespace} {name!r} needs {path} in {allowed}, "
                                    f"got {value!r}"
                                )
                            if len(allowed) == 1:
                                if path in updates and updates[path] != allowed[0]:
                                    raise ValueError(
                                        f"conflicting component REQUIRES for {path}"
                                    )
                                updates[path] = allowed[0]
                    if updates:
                        break
                if updates:
                    break
            if not updates:
                assert experiment is not None
                experiment.__post_init__()
                return experiment
            for path, value in updates.items():
                parent = pending
                parts = path.split(".")
                for part in parts[:-1]:
                    parent = parent.setdefault(part, {})
                parent[parts[-1]] = deepcopy(value)

    @classmethod
    def _from_dict(
        cls, *, values: dict[str, Any], original: dict[str, Any]
    ) -> Experiment:
        allowed = {
            "task",
            "training",
            "data",
            "model",
            "learner",
            "scheduler",
            "initialization",
            "strategy",
            "evaluation",
            "runtime",
            "client_model",
            "groups",
            "clients",
            "reproduces",
        }
        extra = values.keys() - allowed
        if extra:
            raise TypeError(f"unknown experiment keys: {sorted(extra)}")
        task = _resolve(namespace="task", raw=values.get("task", {}))
        data_values = values.get("data", {})
        if not isinstance(data_values, dict):
            raise TypeError("data config must be an object")
        data_allowed = {
            "source",
            "partition",
            "split",
            "scaler",
            "view",
            "store",
            "augmentation",
        }
        extra_data = data_values.keys() - data_allowed
        if extra_data:
            raise TypeError(f"unknown data keys: {sorted(extra_data)}")
        data = DataConfig(
            **{
                key: _resolve(namespace=f"data.{key}", raw=data_values.get(key, {}))
                for key in ("source", "partition", "split", "scaler", "view", "store")
            },
            augmentation=(
                None
                if data_values.get("augmentation") is None
                else _resolve(
                    namespace="data.augmentation", raw=data_values["augmentation"]
                )
            ),
        )
        groups = values.get("groups", [])
        if not isinstance(groups, (list, tuple)):
            raise TypeError("groups must be a list of objects")
        task_type = get(category="tasks", name=task.name)
        if hasattr(task, "input_channels") and not task.input_channels:
            task.input_channels = tuple(data.source.input_channels)
        input_length, output_length, _ = _lengths(data=data)
        derived = {
            # A reconstruction task feeds the model the (masked) target window.
            "input_length": output_length if task_type.RECONSTRUCTS else input_length,
            "output_length": output_length,
            "input_channels": len(data.source.input_channels),
            "output_channels": len(getattr(task, "target_channels", ()))
            or len(data.source.input_channels),
            "calendar_channels": len(getattr(task, "calendar", ())),
            "num_classes": getattr(data.source, "num_classes", None),
            "head": task_type.HEAD,
            "reconstruct": task_type.RECONSTRUCTS,
            "num_quantiles": getattr(task, "num_quantiles", None),
            "target_pattern": getattr(task, "target_pattern", None),
        }

        def resolve_model(*, raw: dict[str, Any]) -> Any:
            # Shape fields the model leaves out come from the data and the task, but
            # only fields its Config declares (Linear has no num_classes).
            model_values = dict(raw)
            model_type = get(
                category="models", name=model_values.get("name", _DEFAULTS["model"])
            )
            model_fields = {field.name for field in fields(model_type.Config)}
            for key, value in derived.items():
                if key in model_fields and value is not None:
                    model_values.setdefault(key, value)
                    if model_values[key] != value:
                        raise ValueError(
                            f"model {key}={model_values[key]!r} does not match the "
                            f"data and task ({value!r})"
                        )
            return _resolve(namespace="model", raw=model_values)

        # The task supplies the default protocol, loss and metrics; a metric goes
        # only to the phases its PHASES allow.
        evaluation_values = dict(values.get("evaluation", {}))
        evaluation_values.setdefault("name", getattr(task_type, "PROTOCOL", "Frozen"))
        evaluation_values.setdefault("loss", task_type.LOSS)
        evaluation_values.setdefault(
            "metrics",
            {
                phase: [
                    metric
                    for metric in task_type.METRICS
                    if phase
                    in info(category="metrics", name=metric).get("PHASES", (phase,))
                ]
                for phase in ("validation", "test")
            },
        )
        return cls(
            task=task,
            training=(
                None
                if values.get("training") is None
                else _resolve(namespace="training", raw=values["training"])
            ),
            data=data,
            model=resolve_model(raw=values.get("model", {})),
            learner=_resolve(namespace="learner", raw=values.get("learner", {})),
            scheduler=_resolve(namespace="scheduler", raw=values.get("scheduler", {})),
            initialization=_resolve(
                namespace="initialization", raw=values.get("initialization", {})
            ),
            strategy=_resolve(namespace="strategy", raw=values.get("strategy", {})),
            evaluation=_resolve(namespace="evaluation", raw=evaluation_values),
            runtime=_resolve(namespace="runtime", raw=values.get("runtime", {})),
            client_model=(
                None
                if values.get("client_model") is None
                else resolve_model(raw=values["client_model"])
            ),
            groups=tuple(
                cls.from_dict(values=_group_values(base=original, group=group))
                for group in groups
            ),
            clients=tuple(values.get("clients", ())),
            reproduces=(
                None
                if values.get("reproduces") is None
                else Reproduces(**values["reproduces"])
            ),
            _validate=False,
        )

    def __post_init__(  # positional: dataclass InitVar.
        self, _validate: bool = True
    ) -> None:
        """Reject losses and metrics that declare tasks excluding this experiment's."""
        if not _validate:
            return
        task = self.task.name
        selected = tuple(_selected_components(experiment=self))
        for namespace, category, name in selected:
            for path, allowed in (
                info(category=category, name=name).get("REQUIRES", {}).items()
            ):
                value = _path_value(experiment=self, path=path)
                if value not in allowed:
                    raise ValueError(
                        f"{namespace} {name!r} needs {path} in {allowed}, got {value!r}"
                    )
            component_type = get(category=category, name=name)
            if hasattr(component_type, "validate"):
                component_type.validate(experiment=self)
        for _, category, name in selected:
            require(category=category, name=name, task=task)
        require(category="data/sources", name=self.data.source.name, task=task)
        require(category="models", name=self.model.name, task=task)
        require(category="strategies", name=self.strategy.name, task=task)
        # On an irregular source NaN marks an unobserved value; a model that does not
        # read masks would train on NaN.
        source_type = get(category="data/sources", name=self.data.source.name)
        if issubclass(source_type, Irregular) and not info(
            category="models", name=self.model.name
        ).get("MASKS"):
            raise ValueError(
                f"model {self.model.name!r} does not read masks; "
                f"{self.data.source.name!r} is irregular, pick a model that "
                "declares MASKS"
            )
        # A target-only model would silently ignore requested covariates.
        if getattr(self.task, "target_channels", ()) or getattr(
            self.task, "calendar", ()
        ):
            if not info(category="models", name=self.model.name).get("COVARIATES"):
                raise ValueError(
                    f"model {self.model.name!r} takes no covariates; drop task "
                    "target_channels/calendar or pick a model that declares COVARIATES"
                )
        # Past covariates need a model that reads them, not only calendar marks.
        if getattr(self.task, "target_channels", ()) and "past" not in info(
            category="models", name=self.model.name
        ).get("COVARIATES", ()):
            raise ValueError(f"model {self.model.name!r} reads no past covariates")
        # A model whose state cannot be averaged (a serialized tree ensemble) names
        # the strategies that never average it.
        strategies = info(category="models", name=self.model.name).get("STRATEGIES")
        if strategies is not None and self.strategy.name not in strategies:
            raise ValueError(
                f"model {self.model.name!r} supports strategies {strategies}, "
                f"not {self.strategy.name!r}"
            )
        # A strategy tied to a model's structure (SplitNN's party blocks) or to
        # datasets names them, or a family class they inherit (`Panel`); its TASKS
        # are checked above.
        strategy = info(category="strategies", name=self.strategy.name)
        models = strategy.get("MODELS")
        if models is not None and not _named(
            component=get(category="models", name=self.model.name), names=models
        ):
            raise ValueError(
                f"strategy {self.strategy.name!r} supports models {models}, "
                f"not {self.model.name!r}"
            )
        sources = strategy.get("SOURCES")
        if sources is not None and not _named(
            component=get(category="data/sources", name=self.data.source.name),
            names=sources,
        ):
            raise ValueError(
                f"strategy {self.strategy.name!r} supports sources {sources}, "
                f"not {self.data.source.name!r}"
            )
        models = info(category="initializers", name=self.initialization.name).get(
            "MODELS"
        )
        if models is not None and self.model.name not in models:
            raise ValueError(
                f"initialization {self.initialization.name!r} is for {models}, "
                f"not {self.model.name!r}"
            )
        if self.client_model is not None:
            require(category="models", name=self.client_model.name, task=task)
            if not info(category="strategies", name=self.strategy.name).get(
                "CLIENT_MODEL"
            ):
                raise ValueError(
                    f"strategy {self.strategy.name!r} trains the shared model on "
                    "every client; client_model needs a strategy that declares "
                    "CLIENT_MODEL"
                )
        finetune_epochs = getattr(self.evaluation, "finetune_epochs", 0)
        if strategy.get("FINETUNE") and finetune_epochs <= 0:
            raise ValueError(
                f"strategy {self.strategy.name!r} is scored after local fine-tuning; "
                "set evaluation.finetune_epochs > 0"
            )
        if finetune_epochs and (
            self.client_model is not None
            or any(member.client_model is not None for member in self.groups)
        ):
            raise ValueError(
                "fine-tuning trains the server model's architecture; it does not "
                "support client_model"
            )
        if (self.groups or self.client_model is not None) and (
            not info(category="evaluation", name=self.evaluation.name).get("GROUPS")
        ):
            raise ValueError(
                "client groups and client_model are scored by the Frozen protocol "
                f"only, not {self.evaluation.name!r}"
            )
        require(category="metrics", name=self.evaluation.loss, task=task, role="loss")
        require(category="evaluation", name=self.evaluation.name, task=task)
        if hasattr(self.evaluation, "adaptation"):
            require(category="adaptation", name=self.evaluation.adaptation, task=task)
        for phase, metric_names in self.evaluation.metrics.items():
            if phase not in ("train", "validation", "test"):
                raise ValueError(f"unknown evaluation phase {phase!r}")
            for name in metric_names:
                require(
                    category="metrics", name=name, task=task, phase=phase, role="metric"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def member(self, *, group: int) -> Experiment:
        """The experiment of one client group; itself when there are no groups."""
        return self.groups[group] if self.groups else self

    def lengths(self) -> tuple[int, int, int]:
        """(input_length, output_length, offset_length) of one example."""
        return _lengths(data=self.data)


def _path_value(*, experiment: Experiment, path: str) -> Any:
    """Read a declared config path; only an absent optional namespace yields None."""
    value: Any = experiment
    for part in path.split("."):
        if value is None:
            return None
        value = getattr(value, part)
    return tuple(value) if isinstance(value, list) else value


def _raw_path_value(*, values: dict[str, Any], path: str) -> Any:
    value: Any = values
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return tuple(value) if isinstance(value, list) else value


def _provided_paths(*, values: dict[str, Any]) -> tuple[set[str], set[str]]:
    provided: set[str] = set()
    leaves: set[str] = set()
    remaining = [("", values)]
    while remaining:
        prefix, mapping = remaining.pop()
        for key, value in mapping.items():
            path = f"{prefix}.{key}" if prefix else key
            provided.add(path)
            if isinstance(value, dict):
                remaining.append((path, value))
            else:
                leaves.add(path)
    return provided, leaves


def _required_names(*, values: dict[str, Any]) -> dict[str, Any]:
    """Select CLI option types without validating a partially applied override set."""
    provided, leaves = _provided_paths(values=values)
    selected = deepcopy(values)
    seen: set[str] = set()
    while True:
        state = json.dumps(selected, sort_keys=True, default=str)
        if state in seen:
            raise ValueError("conflicting component name requirements")
        seen.add(state)
        updates = {}
        for _, category, name in _selected_components(experiment=selected):
            for path, allowed in (
                info(category=category, name=name).get("REQUIRES", {}).items()
            ):
                if not path.endswith(".name") or len(allowed) != 1:
                    continue
                if path in provided or any(
                    path.startswith(key + ".") for key in leaves
                ):
                    continue
                if _raw_path_value(values=selected, path=path) != allowed[0]:
                    if path in updates and updates[path] != allowed[0]:
                        raise ValueError(
                            f"conflicting component name requirements for {path}"
                        )
                    updates[path] = allowed[0]
        if not updates:
            return selected
        for path, value in updates.items():
            parent = selected
            parts = path.split(".")
            for part in parts[:-1]:
                parent = parent.setdefault(part, {})
            parent[parts[-1]] = deepcopy(value)


def _selected_components(
    *, experiment: Experiment | dict[str, Any]
) -> Iterator[tuple[str, str, str]]:
    """Walk component configs and named references, including nested selections."""

    def walk(
        *, value: Any, path: str, category: str | None = None
    ) -> Iterator[tuple[str, str, str]]:
        category = (
            category
            or _CATEGORIES.get(path)
            or _COMPONENT_FIELDS.get(path.rsplit(".", 1)[-1])
        )
        if isinstance(value, dict):
            named = category and ("name" in value or path in _DEFAULTS)
            if named and category is not None:
                name = value.get("name", _DEFAULTS.get(path))
                if name is not None:
                    yield path, category, name
            for key, child in value.items():
                if named and key == "name":
                    continue
                yield from walk(
                    value=child,
                    path=f"{path}.{key}" if path else key,
                    category=None if named else category,
                )
        elif isinstance(value, (tuple, list)) and category:
            for index, child in enumerate(value):
                yield from walk(value=child, path=f"{path}.{index}", category=category)
        elif category and value is not None:
            name = value if isinstance(value, str) else getattr(value, "name", None)
            if name is not None:
                yield path, category, name
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                yield from walk(
                    value=getattr(value, field.name),
                    path=f"{path}.{field.name}" if path else field.name,
                )

    yield from walk(value=experiment, path="")


def _named(*, component: type, names: tuple[str, ...]) -> bool:
    """Whether a component class, or a family class it inherits, is among `names`."""
    return any(cls.__name__ in names for cls in component.__mro__)


def _group_values(*, base: dict[str, Any], group: Any) -> dict[str, Any]:
    """The base values with one group's `data` components, `client_model` and
    `clients`.

    A group component with the base's name (or none) overrides its fields; another
    name replaces it, as a `--namespace.name` override does.
    """
    if not isinstance(group, dict) or group.keys() - {
        "data",
        "client_model",
        "clients",
    }:
        raise TypeError("a group is an object with only data, client_model, clients")
    values = {key: value for key, value in base.items() if key != "groups"}
    data = dict(values.get("data", {}))
    for key, component in group.get("data", {}).items():
        current = data.get(key)
        if isinstance(current, dict) and component.get(
            "name", current.get("name")
        ) == current.get("name"):
            data[key] = {**current, **component}
        else:
            data[key] = component
    values["data"] = data
    for key in ("client_model", "clients"):
        if key in group:
            values[key] = group[key]
    return values


def _lengths(*, data: DataConfig) -> tuple[int, int, int]:
    """Series windows come from the split; whole sequences from the source.

    A split with `window_length` (Predefined on series) gives target windows, as
    reconstruction tasks read them.
    """
    split = data.split
    if hasattr(split, "input_length"):
        return split.input_length, split.output_length, split.offset_length
    if getattr(split, "window_length", 0):
        return 0, split.window_length, 0
    return data.source.input_length, 0, 0


_CATEGORIES = {
    "task": "tasks",
    "training": "training",
    "data.source": "data/sources",
    "data.partition": "data/partitions",
    "data.split": "data/splits",
    "data.scaler": "scalers",
    "data.view": "data/views",
    "data.store": "data/stores",
    "data.augmentation": "augmentations",
    "model": "models",
    "learner": "learners",
    "learner.optimizer": "optimizers",
    "evaluation.optimizer": "optimizers",
    "scheduler": "schedulers",
    "initialization": "initializers",
    "strategy": "strategies",
    "evaluation": "evaluation",
    "runtime": "runtime/executors",
}
_COMPONENT_FIELDS = {
    **{path.rsplit(".", 1)[-1]: category for path, category in _CATEGORIES.items()},
    "client_model": "models",
    "regime": "training",
    "adaptation": "adaptation",
    "topology": "topologies",
    "replay": "replay",
    "member": "data/sources",
    "members": "data/sources",
    "loss": "metrics",
    "metrics": "metrics",
}
_DEFAULTS = {
    "task": "Forecasting",
    "training": None,
    "data.source": "ETDatasetHour",
    "data.partition": "PerFile",
    "data.split": "CompleteWindowRatio",
    "data.scaler": "Standard",
    "data.view": "Offline",
    "data.store": "Parquet",
    "data.augmentation": "Jitter",
    "model": "Linear",
    "learner": "Gradient",
    "learner.optimizer": "Adam",
    "evaluation.optimizer": "Adam",
    "scheduler": "Constant",
    "initialization": "Random",
    "strategy": "FedAvg",
    "evaluation": "Frozen",
    "runtime": "Serial",
}


def _resolve(*, namespace: str, raw: Any) -> Any:
    if not isinstance(raw, dict):
        raise TypeError(f"{namespace} config must be an object")
    values = dict(raw)
    name = values.pop("name", _DEFAULTS[namespace])
    if name is None:
        raise ValueError(f"{namespace}.name must be explicit")
    values["name"] = name
    component = get(category=_CATEGORIES[namespace], name=name)
    config_type = getattr(component, "Config", None)
    if config_type is None:
        if set(values) != {"name"}:
            raise TypeError(
                f"{name} does not accept config keys: "
                f"{sorted(values.keys() - {'name'})}"
            )
        return component
    for key in ("input_channels", "ratios"):
        if key in values and isinstance(values[key], list):
            values[key] = tuple(values[key])
    if "optimizer" in {item.name for item in fields(config_type)}:
        values["optimizer"] = _resolve(
            namespace=f"{namespace}.optimizer", raw=values.get("optimizer", {})
        )
    if isinstance(values.get("metrics"), dict):
        values["metrics"] = {
            phase: tuple(metric_names)
            for phase, metric_names in values["metrics"].items()
        }
    return config_type(**values)
