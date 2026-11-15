"""Set up resolved components and persist one reviewable run directory."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from kairots import __version__
from kairots.config.schema import Experiment
from kairots.data.contracts import ClientRecord
from kairots.data.sources.Composite import Composite
from kairots.data.stores.Parquet import family
from kairots.evaluation.populations import select as select_populations
from kairots.metrics._continual import forecasting_summary
from kairots.registry import get, info
from kairots.runtime.contracts import ClientState, Job, Message
from kairots.runtime.device import execution_device
from kairots.runtime.model_statistics import model_statistics
from kairots.runtime.permissions import DataPermissions
from kairots.runtime.seeds import configure_determinism, derive_seed
from kairots.runtime.source_run import resolve_source
from kairots.runtime.state import StateStore
from kairots.runtime.worker import Worker


@dataclass(kw_only=True)
class RunResult:
    run_dir: str
    rounds: list[dict[str, Any]]
    wall_seconds: float


def _json_default(value: Any) -> Any:  # positional: json encoder callback.
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (Tensor,)):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _same_model_config(*, source: dict[str, Any], target: Any) -> bool:
    target_values = json.loads(json.dumps(asdict(target), default=_json_default))
    return all(
        source.get(key) == value
        for key, value in target_values.items()
        if key != "input_channels"
    )


def _run_path(*, experiment: Experiment) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = json.dumps(experiment.to_dict(), sort_keys=True, default=str).encode()
    suffix = hashlib.sha256(payload + str(time.time_ns()).encode()).hexdigest()[:8]
    experiment_name = (
        f"{experiment.data.source.name}-{experiment.strategy.name}-"
        f"{experiment.model.name}"
    )
    path = Path(experiment.runtime.runs_dir) / experiment_name / f"{stamp}-{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir()
    return path


def _client_id(*, name: str) -> int | None:
    """The client id of a `client:<id>` message party, else None."""
    prefix, _, number = name.partition(":")
    return int(number) if prefix == "client" and number.isdigit() else None


def _git(*args: str) -> str:
    """Output of a git command in the source tree, or '' outside a git checkout."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return done.stdout if done.returncode == 0 else ""


def _provenance() -> dict[str, object]:
    """The code revision and the machine that produced a run."""
    diff = _git("diff", "HEAD")
    return {
        "git_sha": _git("rev-parse", "HEAD").strip(),
        # Tracked changes only; untracked files are not covered.
        "git_dirty": hashlib.sha256(diff.encode()).hexdigest() if diff else "",
        "host": platform.node(),
        "cuda": torch.version.cuda or "",
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def _shapes(*, module: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Parameter and buffer names with their shapes: a model's structure."""
    return tuple(
        (key, tuple(value.shape)) for key, value in module.state_dict().items()
    )


def _fit_record(*, record: ClientRecord, experiment: Experiment) -> ClientRecord:
    """Give fit workers only source/support windows and no later series rows."""
    permissions = DataPermissions(source=("train",), support=("validation",))
    windows = {phase: record.windows.get(phase, ()) for phase in permissions.fit_phases}
    member = experiment.member(group=record.group)
    _, output_length, _ = member.lengths()
    visible_until = None
    if output_length:
        last_start = max(
            (last for ranges in windows.values() for _, last in ranges), default=-1
        )
        visible_until = last_start + record.target_offset + output_length
    return replace(
        record,
        windows=windows,
        permissions=permissions.fit_phases,
        visible_until=visible_until,
    )


def run(*, experiment: Experiment) -> RunResult:
    """Train and evaluate using the phases configured on the experiment."""
    return _execute(experiment=experiment, permission="run")


def fit(*, experiment: Experiment) -> RunResult:
    """Fit a source model using train windows and validation-only selection."""
    if info(category="strategies", name=experiment.strategy.name).get(
        "SOURCE_RUN", False
    ):
        raise ValueError("fit needs a strategy that trains a source model")
    validation_metrics = experiment.evaluation.metrics.get("validation", ())
    metrics = {
        phase: names
        for phase, names in experiment.evaluation.metrics.items()
        if phase == "train" or (phase == "validation" and validation_metrics)
    }
    evaluation = replace(
        experiment.evaluation,
        metrics=metrics,
        selection_metric=(
            experiment.evaluation.selection_metric if validation_metrics else ""
        ),
        leaky_test_selection=False,
    )
    runtime = replace(experiment.runtime, save_checkpoints=True)
    return _execute(
        experiment=replace(experiment, evaluation=evaluation, runtime=runtime),
        permission="fit",
    )


def evaluate(*, experiment: Experiment, source_run: str | Path) -> RunResult:
    """Evaluate a completed fit artifact on query windows without target fitting."""
    path = Path(source_run)
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"source_run is not a completed fit artifact: {path}"
        ) from error
    if (
        summary.get("status") != "completed"
        or manifest.get("data_permissions", {}).get("mode") != "fit"
    ):
        raise ValueError("source_run must be produced by fit() and completed")
    checkpoint = "best_val" if summary.get("best_val_round") is not None else "final"
    if not (path / "checkpoints" / f"{checkpoint}.pt").is_file():
        raise ValueError(f"source_run is missing its {checkpoint!r} checkpoint")
    strategy_type = get(category="strategies", name="ZeroShot")
    initializer_type = get(category="initializers", name="Pretrained")
    evaluation_changes = {
        "metrics": {"test": experiment.evaluation.metrics.get("test", ())},
        "patience": 0,
        "selection_metric": "",
        "leaky_test_selection": False,
    }
    if hasattr(experiment.evaluation, "finetune_epochs"):
        evaluation_changes["finetune_epochs"] = 0
    evaluation = replace(experiment.evaluation, **evaluation_changes)
    return _execute(
        experiment=replace(
            experiment,
            strategy=strategy_type.Config(),
            initialization=initializer_type.Config(
                run_dir=str(path), checkpoint=checkpoint
            ),
            evaluation=evaluation,
        ),
        permission="evaluate",
    )


scoring_configs = {
    "generalization": lambda *, member: asdict(member.model),
    "personalization": lambda *, member: asdict(member.client_model or member.model),
}


def _seen_windows(
    *, regime: Any, windows: list[Any], period: int, num_classes: int | None
) -> list[Any]:
    """Windows of tasks 0..period, in their original order."""
    seen = {
        item
        for task in range(period + 1)
        for item in regime.windows(
            windows=windows, period=task, num_classes=num_classes
        )
    }
    return [item for item in windows if item in seen]


def _execute(
    *, experiment: Experiment, permission: Literal["run", "fit", "evaluate"]
) -> RunResult:
    started = time.perf_counter()
    source_run = info(category="strategies", name=experiment.strategy.name).get(
        "SOURCE_RUN", False
    )
    if source_run and experiment.strategy.source_config:
        experiment = resolve_source(experiment=experiment)
    resume = experiment.runtime.resume
    run_dir = Path(resume) if resume else _run_path(experiment=experiment)
    resume_path = run_dir / "checkpoints" / "resume.pt"
    resolved = json.loads(json.dumps(experiment.to_dict(), default=_json_default))
    if resume:
        # A resume continues the same run: refuse another config before writing.
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        saved = saved["resolved"]
        saved["runtime"]["resume"] = resolved["runtime"]["resume"]
        changed = sorted(key for key in resolved if resolved[key] != saved.get(key))
        if changed:
            raise ValueError(f"resume config differs from the run's in {changed}")
    config_data = {
        "resolved": resolved,
        "seed": experiment.runtime.seed,
        "versions": {
            "kairots": __version__,
            "torch": torch.__version__,
            "ray": __import__("ray").__version__,
            "python": platform.python_version(),
            **_provenance(),
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_data, indent=2, default=_json_default), encoding="utf-8"
    )
    (run_dir / "rounds.jsonl").touch()
    (run_dir / "clients.jsonl").touch()
    data_permissions = {
        "run": DataPermissions(
            source=("train",), support=("validation",), query=("test",)
        ),
        "fit": DataPermissions(source=("train",), support=("validation",)),
        "evaluate": DataPermissions(query=("test",)),
    }[permission]
    manifest: dict[str, Any] = {
        "raw_files": [],
        "channel_order": list(experiment.data.source.input_channels),
        "clients": [],
        "exclusions": [],
        "data_permissions": {
            "mode": permission,
            "source": data_permissions.source,
            "support": data_permissions.support,
            "query": data_permissions.query,
            "query_labels": (
                "unavailable during fit; metric-only or post-prediction otherwise"
            ),
            "source_run": experiment.initialization.run_dir
            if permission == "evaluate"
            else None,
        },
    }
    rounds: list[dict[str, Any]] = []
    total_bytes = 0
    best_score = float("inf")
    selection_phase = (
        "test" if experiment.evaluation.leaky_test_selection else "validation"
    )
    selection_name = (
        experiment.evaluation.selection_metric or experiment.evaluation.loss
    )
    if experiment.evaluation.selection_metric and (
        selection_name not in experiment.evaluation.metrics.get(selection_phase, ())
    ):
        raise ValueError(
            f"evaluation.metrics.{selection_phase} must list {selection_name!r}, "
            "the metric that selects the checkpoint"
        )
    selection_direction = (
        info(category="metrics", name=selection_name).get("DIRECTION", "min")
        if selection_name in experiment.evaluation.metrics.get(selection_phase, ())
        else "min"
    )
    best_round: int | None = None
    best_metrics: dict[str, float] = {}
    executor: Any = None
    status = "failed"
    failure = "run failed"
    unseen: set[int] = set()
    try:
        if experiment.groups:
            # Each group prepares its own clients (own cuts and train-fitted
            # scalers) and keeps `clients` of them; run client identifiers number them
            # clients in group order, and each entry keeps its partition id.
            records = []
            entries: list[dict[str, Any]] = []
            exclusions: list[dict[str, Any]] = []
            groups: list[dict[str, Any]] = []
            for group, member in enumerate(experiment.groups):
                store_type = get(category="data/stores", name=member.data.store.name)
                group_manifest, group_records = store_type(
                    config=member.data.store
                ).prepare(experiment=member)
                client_id_map = {
                    record.client_id: len(records) + index
                    for index, record in enumerate(
                        record
                        for record in group_records
                        if not member.clients or record.client_id in member.clients
                    )
                }
                split_type = get(category="data/splits", name=member.data.split.name)
                partition_unseen = (
                    split_type(config=member.data.split).unseen(
                        client_ids=list(client_id_map)
                    )
                    if hasattr(split_type, "unseen")
                    else []
                )
                group_unseen = sorted(
                    client_id_map[client_id] for client_id in partition_unseen
                )
                group_manifest["unseen_clients"] = group_unseen
                group_manifest["unseen_partition_clients"] = partition_unseen
                unseen.update(group_unseen)
                records += [
                    replace(
                        record, client_id=client_id_map[record.client_id], group=group
                    )
                    for record in group_records
                    if record.client_id in client_id_map
                ]
                entries += [
                    {
                        **entry,
                        "client_id": client_id_map[entry["client_id"]],
                        "group": group,
                        "partition_client_id": entry["client_id"],
                    }
                    for entry in group_manifest.pop("clients")
                    if entry["client_id"] in client_id_map
                ]
                exclusions += [
                    {**entry, "group": group}
                    for entry in group_manifest.pop("exclusions")
                ]
                groups.append(group_manifest)
            manifest.update(groups=groups, clients=entries, exclusions=exclusions)
        else:
            store_type = get(category="data/stores", name=experiment.data.store.name)
            store_manifest, records = store_type(config=experiment.data.store).prepare(
                experiment=experiment
            )
            manifest.update(store_manifest)
        if permission == "fit":
            records = [
                _fit_record(record=record, experiment=experiment) for record in records
            ]
        manifest["scaler_parameters"] = [
            {"client_id": entry["client_id"], **entry["scaler"]}
            for entry in manifest["clients"]
        ]
        source_type = get(category="data/sources", name=experiment.data.source.name)
        members = getattr(source_type, "MEMBERS", ())
        families = {family(source_type=member) for member in members or (source_type,)}
        manifest["dataset_family"] = (
            next(iter(families))
            if not experiment.groups
            and len(families) == 1
            and (not issubclass(source_type, Composite) or members)
            else None
        )
        if resume:
            # The saved parameters only fit the data the run prepared.
            saved_manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            prepared = json.loads(json.dumps(manifest, default=_json_default))
            if any(saved_manifest.get(key) != value for key, value in prepared.items()):
                raise ValueError("prepared data changed since the run started")
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
        )
        clients = [(record.client_id, record.name, record) for record in records]
        if not clients:
            raise ValueError("no clients meet the configured minimum window counts")
        model_type = get(category="models", name=experiment.model.name)
        learner_type = get(category="learners", name=experiment.learner.name)
        strategy_type = get(category="strategies", name=experiment.strategy.name)
        task_type = get(category="tasks", name=experiment.task.name)
        view_type = get(category="data/views", name=experiment.data.view.name)
        evaluation_type = get(category="evaluation", name=experiment.evaluation.name)
        loss_type = get(category="metrics", name=experiment.evaluation.loss)
        strategy = strategy_type(config=experiment.strategy)
        worker = Worker(
            experiment=experiment,
            strategy=strategy,
            task=task_type(config=experiment.task),
            learner_type=learner_type,
            loss_function=loss_type(config=loss_type.Config()),
            view=view_type(config=experiment.data.view),
        )
        executor_type = get(category="runtime/executors", name=experiment.runtime.name)
        executor = executor_type(experiment=experiment, worker=worker)
        # The orchestrator loads every client to evaluate; move evaluation
        # into worker jobs before runs with thousands of clients.
        loaded = {
            record.client_id: get(category="data/stores", name=record.store).load(
                record=record
            )
            for record in records
        }
        group_of = {record.client_id: record.group for record in records}
        # Each client's train-fitted scaler, passed only to inverse scoring.
        scales = {
            record.client_id: (torch.tensor(record.mean), torch.tensor(record.scale))
            for record in records
        }
        inverse = getattr(experiment.evaluation, "inverse", False)
        configure_determinism(deterministic=experiment.runtime.deterministic)
        torch.manual_seed(experiment.runtime.seed)
        # Held-out clients (a split with `unseen`) never train; they are scored on
        # test as the `unseen` population.
        split_type = get(category="data/splits", name=experiment.data.split.name)
        if not experiment.groups:
            unseen = (
                set(
                    split_type(config=experiment.data.split).unseen(
                        client_ids=[client_id for client_id, _, _ in clients]
                    )
                )
                if hasattr(split_type, "unseen")
                else set()
            )
        manifest["unseen_clients"] = sorted(unseen)
        # Workers never hold a held-out client, so no job (pooled ones included)
        # can train on it.
        for client_id, _, record in clients:
            if client_id not in unseen:
                executor.cache_client(client_id=client_id, data=record)
        model = model_type(config=experiment.model)
        # Once, on the canonical global model; clients receive its values.
        policy_type = get(category="initializers", name=experiment.initialization.name)
        manifest["initialization"] = {
            "policy": experiment.initialization.name,
            "seed": experiment.runtime.seed,
            "parameters": policy_type(config=experiment.initialization).apply(
                model=model,
                manifest=manifest,
            ),
        }
        if source_run:
            source_path = Path(manifest["pretraining_calendar_check"]["source_run"])
            source_model = json.loads(
                (source_path / "config.json").read_text(encoding="utf-8")
            )["resolved"]["model"]
            if not _same_model_config(source=source_model, target=experiment.model):
                raise ValueError(
                    "ZeroShot requires matching source and target model configs"
                )
            if set(manifest["initialization"]["parameters"]) != set(model.state_dict()):
                raise ValueError(
                    "ZeroShot requires every checkpoint parameter and buffer"
                )
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
        )
        kept: dict[int, dict[str, Tensor]] = {}
        global_params = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        # Scoring models per (group, scoring): the server model on the group's data
        # for generalization, the client model for personalization. Built off the
        # global RNG stream, so groups leave the run's draws unchanged.
        members = experiment.groups or (experiment,)
        scoring_models: dict[tuple[int, str], torch.nn.Module] = {}
        with torch.random.fork_rng(devices=[]):
            for group, member in enumerate(members):
                for scoring, config in (
                    ("generalization", member.model),
                    ("personalization", member.client_model or member.model),
                ):
                    scoring_models[group, scoring] = (
                        model
                        if config is experiment.model
                        else get(category="models", name=config.name)(config=config)
                    )
        for group in range(len(members)):
            if _shapes(module=scoring_models[group, "generalization"]) != _shapes(
                module=model
            ):
                raise ValueError(
                    f"group {group}'s server model has other parameter shapes than "
                    "the global model (it depends on the group's channels or "
                    "lengths); give the groups matching data shapes"
                )
        # Clients whose scoring models share parameter names and shapes form a
        # cohort (one architecture, width and depth); each is also scored alone.
        cohorts: dict[str, list[list[int]]] = {}
        for scoring in ("generalization", "personalization"):
            signatures: dict[tuple[tuple[str, tuple[int, ...]], ...], list[int]] = {}
            for record in records:
                signatures.setdefault(
                    _shapes(module=scoring_models[record.group, scoring]), []
                ).append(record.client_id)
            cohorts[scoring] = list(signatures.values())
        heterogeneous = bool(experiment.groups) or experiment.client_model is not None
        if heterogeneous:
            manifest["cohorts"] = {
                scoring: [
                    {
                        "model": scoring_configs[scoring](
                            member=members[group_of[client_ids[0]]]
                        ),
                        "clients": client_ids,
                    }
                    for client_ids in scoring_cohorts
                ]
                for scoring, scoring_cohorts in cohorts.items()
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
            )
        state_store = StateStore(
            num_clients=max(client_id for client_id, _, _ in clients) + 1
        )
        for client_id, _, _ in clients:
            state_store.clients[client_id] = strategy.initial_state(
                global_params=global_params
            )
        evaluator = evaluation_type(
            config=experiment.evaluation,
            loss_function=loss_type(config=loss_type.Config()),
        )
        # Validation runs when the split has it (it then picks the best checkpoint);
        # train only on request. Without validation the final round is reported.
        has_validation = any(client[1]["validation"] for client in loaded.values())
        if experiment.evaluation.leaky_test_selection and has_validation:
            raise ValueError(
                "evaluation.leaky_test_selection selects on the test fold, so the "
                "split must have no validation fold"
            )
        if permission == "fit":
            phases = [
                phase
                for phase in ("train", "validation")
                if phase in experiment.evaluation.metrics
                and (phase != "validation" or has_validation)
            ]
        elif permission == "evaluate":
            phases = ["test"]
        else:
            phases = [
                phase
                for phase in ("train", "validation", "test")
                if (phase != "train" or phase in experiment.evaluation.metrics)
                and (phase != "validation" or has_validation)
            ]
        prefixes = {"train": "train", "validation": "val", "test": "test"}
        view = view_type(config=experiment.data.view)
        # One training batch shape of the first group-0 trainable client; the
        # counts depend on shapes only.
        sample_id = next(
            client_id
            for client_id, _, record in clients
            if client_id not in unseen and record.group == 0
        )
        input_length, output_length, offset_length = members[0].lengths()
        sample_values, sample_splits = loaded[sample_id]
        sample_x, sample_y = next(
            get(category="data/views", name=members[0].data.view.name)(
                config=members[0].data.view
            ).batches(
                values=sample_values,
                windows=sample_splits["train"],
                input_length=input_length,
                output_length=output_length,
                offset_length=offset_length,
                batch_size=members[0].data.source.batch_size,
                shuffle=False,
                generator=torch.Generator(),
            )
        )
        with torch.random.fork_rng(devices=[]):
            sample_x, _ = get(category="tasks", name=members[0].task.name)(
                config=members[0].task
            ).prepare(
                batch_x=sample_x,
                batch_y=sample_y,
                output_length=output_length,
                generator=torch.Generator(),
            )
            statistics = model_statistics(model=model, batch_x=sample_x)
        first_iteration = 0
        if resume:
            saved = torch.load(resume_path, weights_only=False)
            if saved.get("selection_direction", "min") != selection_direction:
                raise ValueError(
                    "resume used a different validation direction; restart run"
                )
            first_iteration = saved["current_iter"] + 1
            global_params, kept = saved["global_params"], saved["kept"]
            state_store.clients = saved["clients"]
            state_store.accepted_operations = saved["accepted_ops"]
            rounds, total_bytes = saved["rounds"], saved["total_bytes"]
            best_score, best_round, best_metrics = saved["best"]
            torch.set_rng_state(saved["rng"])
            vars(strategy).update(saved["strategy"])
            # Rows of the interrupted round are written again.
            for name in ("rounds.jsonl", "clients.jsonl"):
                lines = (run_dir / name).read_text(encoding="utf-8").splitlines(True)
                (run_dir / name).write_text(
                    "".join(
                        line
                        for line in lines
                        if json.loads(line)["current_iter"] < first_iteration
                    ),
                    encoding="utf-8",
                )
        training_type = (
            None
            if experiment.training is None
            else get(category="training", name=experiment.training.name)
        )
        regime = (
            None
            if training_type is None or not getattr(training_type, "CONTINUAL", False)
            else training_type(config=experiment.training)
        )
        periods = 1 if regime is None else regime.config.periods
        total_iterations = periods * experiment.strategy.iterations
        remote_executor = info(
            category="runtime/executors", name=experiment.runtime.name
        ).get("REMOTE", False)

        def stage_total(*, values: list[float]) -> float:
            return max(values, default=0.0) if remote_executor else sum(values)

        for current_iteration in range(first_iteration, total_iterations):
            round_started = time.perf_counter()
            selected = strategy.select_clients(
                client_ids=[
                    client_id for client_id, _, _ in clients if client_id not in unseen
                ],
                current_iteration=current_iteration,
                seed=derive_seed(
                    seed=experiment.runtime.seed,
                    current_iteration=current_iteration,
                    client_id=-1,
                    operation="select",
                ),
            )
            jobs: list[Job] = []
            downlink_bytes = 0
            for client_id in selected:
                downlink = strategy.build_downlink(
                    global_params=global_params, client_id=client_id
                )
                downlink_bytes += downlink.billed_bytes
                jobs.append(
                    Job(
                        operation_id=(current_iteration, client_id, "fit"),
                        operation="fit",
                        downlink=downlink,
                        state=state_store.clients[client_id],
                        seed=derive_seed(
                            seed=experiment.runtime.seed,
                            current_iteration=current_iteration,
                            client_id=client_id,
                            operation="fit",
                        ),
                        client_id=client_id,
                    )
                )
            execution_started = time.perf_counter()
            results = executor.execute(jobs=jobs, broadcast=global_params)
            execution_seconds = time.perf_counter() - execution_started
            worker_times = [
                result.timing.get("worker_seconds", 0.0) for result in results
            ]
            worker_seconds = stage_total(values=worker_times)
            data_seconds = stage_total(
                values=[result.timing.get("data_seconds", 0.0) for result in results],
            )
            fit_seconds = stage_total(
                values=[result.timing.get("fit_seconds", 0.0) for result in results],
            )
            aggregate_started = time.perf_counter()
            peer_messages = strategy.exchange(results=results)
            for result in results:
                state_store.accept(result=result)
            # The server sees only what crossed the network: the uplink messages.
            uplinks = [result.uplink for result in results]
            global_params = strategy.aggregate(
                uplinks=uplinks, global_params=global_params
            )
            broadcast = strategy.broadcast(uplinks=uplinks, global_params=global_params)
            downlink_bytes += sum(message.billed_bytes for message in broadcast)
            aggregate_seconds = time.perf_counter() - aggregate_started
            parameters = [
                strategy.client_parameters(
                    global_params=global_params, state=state_store.clients[client_id]
                )
                for client_id in loaded
            ]
            # Checkpoint a client's parameters only where they are its own (not the
            # global tensors themselves).
            kept = {
                client_id: client_parameters
                for client_id, client_parameters in zip(loaded, parameters)
                if any(
                    value is not global_params.get(key)
                    for key, value in client_parameters.items()
                )
            }
            device = execution_device(
                policy=experiment.runtime.device,
                remote=info(
                    category="runtime/executors", name=experiment.runtime.name
                ).get("REMOTE", False),
                num_gpus=experiment.runtime.num_gpus,
            )
            # Generalization scores the server model on every client; a strategy
            # that keeps parameters of its clients' own (personal, local or client
            # models) is also scored with them, as personalization.
            scorings = {"generalization": [global_params] * len(parameters)}
            if kept:
                scorings["personalization"] = parameters
                if regime is not None:
                    scorings["client_generalization"] = parameters
            evaluation_started = time.perf_counter()
            # Each seen client fine-tunes a copy of what it is scored with, on its
            # train windows, before the final round is scored; nothing is sent.
            finetune_epochs = getattr(experiment.evaluation, "finetune_epochs", 0)
            if finetune_epochs and current_iteration == total_iterations - 1:
                tuned_ids = [
                    client_id for client_id in loaded if client_id not in unseen
                ]
                tuned = executor.execute(
                    jobs=[
                        Job(
                            operation_id=(current_iteration, client_id, "finetune"),
                            operation="finetune",
                            downlink=Message(
                                sender=f"client:{client_id}",
                                receiver=f"client:{client_id}",
                                tensors=client_parameters,
                            ),
                            state=ClientState(),
                            seed=derive_seed(
                                seed=experiment.runtime.seed,
                                current_iteration=current_iteration,
                                client_id=client_id,
                                operation="finetune",
                            ),
                            client_id=client_id,
                        )
                        for client_id, client_parameters in zip(loaded, parameters)
                        if client_id in tuned_ids
                    ],
                    broadcast=global_params,
                )
                finetuned = {
                    client_id: {**result.uplink.tensors, **result.state.model_state}
                    for client_id, result in zip(tuned_ids, tuned, strict=True)
                }
                scorings["finetuned"] = [
                    finetuned.get(client_id, client_parameters)
                    for client_id, client_parameters in zip(loaded, parameters)
                ]
            # Population label prefix -> (member client identifiers, phases scored).
            populations = select_populations(
                client_ids=list(loaded),
                unseen_client_ids=unseen,
                phases=phases,
                cross_dataset=source_run,
            )
            position = {client_id: index for index, client_id in enumerate(loaded)}
            evaluated: dict[str, dict[str, float]] = {}
            client_rows: dict[int, dict[str, dict[str, float]]] = {
                client_id: {} for client_id in loaded
            }
            unique_models = {id(module): module for module in scoring_models.values()}
            for module in unique_models.values():
                module.to(device)
            for label, (member_ids, population_phases) in populations.items():
                for scoring, scoring_parameters in scorings.items():
                    if label and scoring == "finetuned":
                        continue
                    # A fine-tuned client keeps the server model's architecture.
                    structure = (
                        "generalization"
                        if scoring == "finetuned"
                        else "personalization"
                        if scoring == "client_generalization"
                        else scoring
                    )
                    # Every member, then each cohort alone when there are several.
                    subsets = {"": member_ids}
                    if len(cohorts[structure]) > 1:
                        subsets.update(
                            {
                                f"cohort{index}_": [
                                    client_id
                                    for client_id in member_ids
                                    if client_id in cohort
                                ]
                                for index, cohort in enumerate(cohorts[structure])
                            }
                        )
                    for part, subset_ids in subsets.items():
                        if not subset_ids:
                            continue
                        for phase in population_phases:
                            phase_metrics, phase_rows = evaluator.evaluate(
                                phase=phase,
                                model=model,
                                clients=[
                                    loaded[client_id]
                                    if regime is None
                                    or (
                                        scoring != "personalization" and phase == "test"
                                    )
                                    else (
                                        loaded[client_id][0],
                                        {
                                            name: _seen_windows(
                                                regime=regime,
                                                windows=windows,
                                                period=current_iteration
                                                // experiment.strategy.iterations,
                                                num_classes=getattr(
                                                    experiment.data.source,
                                                    "num_classes",
                                                    None,
                                                ),
                                            )
                                            for name, windows in loaded[client_id][
                                                1
                                            ].items()
                                        },
                                    )
                                    for client_id in subset_ids
                                ],
                                parameters=[
                                    scoring_parameters[position[client_id]]
                                    for client_id in subset_ids
                                ],
                                **(
                                    {
                                        "scales": [
                                            scales[client_id]
                                            for client_id in subset_ids
                                        ]
                                    }
                                    if inverse
                                    else {}
                                ),
                                **(
                                    {
                                        "models": [
                                            scoring_models[
                                                group_of[client_id], structure
                                            ]
                                            for client_id in subset_ids
                                        ],
                                        "experiments": [
                                            members[group_of[client_id]]
                                            for client_id in subset_ids
                                        ],
                                    }
                                    if heterogeneous
                                    else {}
                                ),
                                experiment=experiment,
                                view=view,
                                task=task_type(config=experiment.task),
                                device=device,
                            )
                            evaluated[f"{label}{scoring}_{part}{prefixes[phase]}"] = (
                                phase_metrics
                            )
                            if part:
                                continue
                            for client_id, client_row in zip(
                                subset_ids, phase_rows, strict=True
                            ):
                                client_rows[client_id][
                                    f"{scoring}_{prefixes[phase]}"
                                ] = client_row
            if regime is not None:
                period = current_iteration // experiment.strategy.iterations
                for seen_period in range(period + 1):
                    for label, period_parameters in (
                        ("", [global_params] * len(loaded)),
                        *(([("local_", parameters)]) if kept else []),
                    ):
                        period_metrics, period_rows = evaluator.evaluate(
                            phase="test",
                            model=model,
                            clients=[
                                (
                                    values,
                                    {
                                        **splits,
                                        "test": regime.windows(
                                            windows=splits["test"],
                                            period=seen_period,
                                            num_classes=getattr(
                                                experiment.data.source,
                                                "num_classes",
                                                None,
                                            ),
                                        ),
                                    },
                                )
                                for values, splits in loaded.values()
                            ],
                            parameters=period_parameters,
                            experiment=experiment,
                            view=view,
                            task=task_type(config=experiment.task),
                            device=device,
                        )
                        evaluated[f"{label}period_{seen_period}"] = period_metrics
                        for client_id, period_row in zip(
                            loaded, period_rows, strict=True
                        ):
                            client_rows[client_id][f"{label}period_{seen_period}"] = (
                                period_row
                            )
            for module in unique_models.values():
                module.to("cpu")
            evaluation_seconds = time.perf_counter() - evaluation_started
            uplink_bytes = sum(result.uplink.billed_bytes for result in results)
            peer_bytes = sum(message.billed_bytes for message in peer_messages)
            # Per-client traffic; messages between other parties (SplitNN's feature
            # and label parties) count only in the round totals.
            traffic = {
                client_id: {"downlink_bytes": 0, "uplink_bytes": 0, "peer_bytes": 0}
                for client_id, _, _ in clients
            }
            for job in jobs:
                traffic[job.client_id]["downlink_bytes"] += job.downlink.billed_bytes
            for result in results:
                traffic[result.operation_id[1]]["uplink_bytes"] += (
                    result.uplink.billed_bytes
                )
            for kind, messages, party in (
                ("downlink_bytes", broadcast, "receiver"),
                ("peer_bytes", peer_messages, "sender"),
            ):
                for message in messages:
                    client_id = _client_id(name=getattr(message, party))
                    if client_id in traffic:
                        traffic[client_id][kind] += message.billed_bytes
            total_bytes += downlink_bytes + uplink_bytes + peer_bytes
            row = {
                "current_iter": current_iteration,
                "selected_clients": selected,
                **(
                    {"period": current_iteration // experiment.strategy.iterations}
                    if regime is not None
                    else {}
                ),
                **{
                    f"{label}_{key}": value
                    for label, phase_metrics in evaluated.items()
                    for key, value in phase_metrics.items()
                },
                "downlink_bytes": downlink_bytes,
                "uplink_bytes": uplink_bytes,
                "peer_bytes": peer_bytes,
                "communication_bytes": downlink_bytes + uplink_bytes + peer_bytes,
                "data_seconds": data_seconds,
                "fit_seconds": fit_seconds,
                "eval_seconds": evaluation_seconds,
                "aggregate_seconds": aggregate_seconds,
                # Includes queueing and executor overhead; Ray overlaps client work.
                "transport_seconds": max(0.0, execution_seconds - worker_seconds),
                "wall_seconds": time.perf_counter() - round_started,
            }
            rounds.append(row)
            with (run_dir / "rounds.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, default=_json_default) + "\n")
            result_by_id = {result.operation_id[1]: result for result in results}
            with (run_dir / "clients.jsonl").open("a", encoding="utf-8") as output:
                for client_id, _, _ in clients:
                    client_result = result_by_id.get(client_id)
                    client_row = {
                        "current_iter": current_iteration,
                        "client_id": client_id,
                        "train_samples": (
                            client_result.train_samples if client_result else 0
                        ),
                        "train_loss": client_result.metrics.get("train_loss")
                        if client_result
                        else None,
                        "timing": client_result.timing if client_result else {},
                        **traffic[client_id],
                        **client_rows[client_id],
                    }
                    output.write(json.dumps(client_row, default=_json_default) + "\n")
            # The best round is picked on validation of what the clients are scored
            # with: personalization when reported, else generalization.
            suffix = prefixes[selection_phase]
            validation = evaluated.get(
                f"personalization_{suffix}",
                evaluated.get(f"generalization_{suffix}", {}),
            )
            score = validation.get(
                f"pooled_{selection_name.lower()}",
                validation.get(f"mean_client_{selection_name.lower()}"),
            )
            # A same-named metric replaces the loss row with its reported quantity.
            if score is not None and selection_direction == "max":
                score = -score
            # A named selection metric keeps TSLib's EarlyStopping rule: a tie with
            # the best so far counts as an improvement, so the later round is kept.
            if score is not None and (
                score <= best_score
                if experiment.evaluation.selection_metric
                else score < best_score
            ):
                best_score, best_round, best_metrics = score, current_iteration, row
                if experiment.runtime.save_checkpoints:
                    torch.save(global_params, run_dir / "checkpoints" / "best_val.pt")
                    if kept:
                        torch.save(
                            kept, run_dir / "checkpoints" / "best_val_clients.pt"
                        )
            if experiment.runtime.save_checkpoints:
                partial = resume_path.with_suffix(".part")
                torch.save(
                    {
                        "current_iter": current_iteration,
                        "global_params": global_params,
                        "kept": kept,
                        "clients": state_store.clients,
                        "accepted_ops": state_store.accepted_operations,
                        "rounds": rounds,
                        "total_bytes": total_bytes,
                        "best": (best_score, best_round, best_metrics),
                        "selection_direction": selection_direction,
                        "rng": torch.get_rng_state(),
                        # Server-side method state (a control variate).
                        "strategy": vars(strategy),
                    },
                    partial,
                )
                partial.replace(resume_path)
            patience = experiment.evaluation.patience
            if (
                patience
                and best_round is not None
                and current_iteration - best_round >= patience
            ):
                break
        if experiment.runtime.save_checkpoints:
            torch.save(global_params, run_dir / "checkpoints" / "final.pt")
            if kept:
                torch.save(kept, run_dir / "checkpoints" / "final_clients.pt")
            # Only an interrupted run needs its resume state.
            resume_path.unlink(missing_ok=True)
        if regime is not None and getattr(regime, "ERROR_MATRIX", False):
            loss_name = experiment.evaluation.loss.lower()
            error = [
                [
                    row[f"period_{evaluated_period}_pooled_{loss_name}"] ** 0.5
                    for evaluated_period in range(row["period"] + 1)
                ]
                for row in rounds
                if (row["current_iter"] + 1) % experiment.strategy.iterations == 0
            ]
            (run_dir / "continual.json").write_text(
                json.dumps({"P": error, **forecasting_summary(matrix=error)}, indent=2),
                encoding="utf-8",
            )
        elif regime is not None:
            boundaries = [
                row
                for row in rounds
                if (row["current_iter"] + 1) % experiment.strategy.iterations == 0
            ]
            matrix = [
                [
                    row[f"period_{evaluated_period}_pooled_accuracy"]
                    for evaluated_period in range(row["period"] + 1)
                ]
                for row in boundaries
            ]
            paper_matrix = [
                [
                    sum(
                        row[f"period_{evaluated_period}_pooled_accuracy"]
                        for row in rounds
                        if row["period"] == period
                    )
                    / experiment.strategy.iterations
                    for evaluated_period in range(period + 1)
                ]
                for period in range(periods)
            ]
            forgetting = [
                max(
                    matrix[learned_period][evaluated_period]
                    for learned_period in range(evaluated_period, len(matrix) - 1)
                )
                - matrix[-1][evaluated_period]
                for evaluated_period in range(len(matrix) - 1)
            ]
            (run_dir / "continual.json").write_text(
                json.dumps(
                    {
                        "R": matrix,
                        "paper_round_average_R": paper_matrix,
                        "average_accuracy": sum(matrix[-1]) / len(matrix[-1]),
                        "forgetting": sum(forgetting) / len(forgetting)
                        if forgetting
                        else 0.0,
                        "server_general_accuracy": sum(
                            row["generalization_test_pooled_accuracy"] for row in rounds
                        )
                        / len(rounds),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        status = "completed"
    except Exception as error:
        # A refused resume keeps the run's own manifest.
        if not resume:
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
            )
        failure = f"{type(error).__name__}: {error}"
    finally:
        if executor is not None:
            try:
                executor.close()
            except Exception as error:
                status = "failed"
                failure = f"executor shutdown failed: {type(error).__name__}: {error}"
        summary = {
            "status": status,
            "final_metrics": rounds[-1] if rounds else {},
            "best_val_round": best_round,
            "best_val_metrics": best_metrics,
            "selection": {
                "phase": selection_phase,
                "metric": selection_name,
                "leaky_test_selection": selection_phase == "test",
            },
            "total_time_seconds": time.perf_counter() - started,
            "total_bytes": total_bytes,
            "model": locals().get("statistics", {}),
        }
        if status == "failed":
            summary["error"] = locals().get("failure", "run failed")
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )
    if status == "failed":
        raise RuntimeError(f"run failed; see {run_dir / 'summary.json'}: {failure}")
    return RunResult(
        run_dir=str(run_dir), rounds=rounds, wall_seconds=time.perf_counter() - started
    )
