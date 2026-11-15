"""Parquet store: prepare each client once, then read only the clients a process needs.

Layout under `root` (`<Family>` is the source's family directory, e.g. ETDataset)::

    <Family>/raw/...                          original files, read-only
    <Family>/prepared/<data_key>/
        manifest.json                         raw files, channels, clients
        clients/<client_id>.parquet           date (series) or sample, step, fold,
                                              label (panel) + channel values
        splits/<split_key>/summary.json       counts, cuts, scalers, window ranges

`data_key` hashes the source and partition configs, each raw file's path, size and
mtime, and the code of the source, partition and store classes (with their bases);
`split_key` adds the split and scaler configs and code. A changed input or changed
code gives a new key, so a prepared directory is never rewritten. Directories are
built in a temporary sibling and renamed into place, so an interrupted build leaves
no partial cache.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import torch

from kairots.data.calendar import features
from kairots.data.contracts import ClientRecord
from kairots.registry import get, info

FORMAT_VERSION = 3
SPLITS = ("train", "validation", "test")


def _key(*, payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def code(*, types: tuple[type, ...]) -> str:
    """Digest of the source files that define these classes, their bases, their
    `Config` classes and, for a composite source, its members (transitively)."""
    pending, kinds = list(types), set()
    while pending:
        kind = pending.pop()
        if kind in kinds:
            continue
        kinds.add(kind)
        pending.extend(getattr(kind, "MEMBERS", ()))
        if isinstance(getattr(kind, "Config", None), type):
            pending.append(kind.Config)
    files = sorted(
        {
            inspect.getfile(base)
            for kind in kinds
            for base in kind.__mro__
            if base.__module__ != "builtins"
        }
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:16]


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:  # positional: json encoder callback.
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def family(*, source_type: type) -> str:
    """Directory that groups a source's raw and prepared data: its family package."""
    parts = source_type.__module__.split(".")
    return parts[parts.index("sources") + 1]


def ranges(*, starts: list[int]) -> list[list[int]]:
    """Collapse sorted window starts into inclusive [first, last] runs."""
    runs: list[list[int]] = []
    for start in starts:
        if runs and start == runs[-1][1] + 1:
            runs[-1][1] = start
        else:
            runs.append([start, start])
    return runs


def _publish(*, build: Path, target: Path) -> None:
    """Rename a finished build into place; a concurrent winner's copy is kept."""
    try:
        os.replace(build, target)
    except OSError:
        if not target.exists():
            raise
        shutil.rmtree(build, ignore_errors=True)


@dataclass(kw_only=True)
class Parquet:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Parquet"
        root: str = "datasets"

    config: Config

    def prepare(self, *, experiment: Any) -> tuple[dict[str, Any], list[ClientRecord]]:
        """Build or reuse the prepared data; return the run manifest and records."""
        data = experiment.data
        source_type = get(category="data/sources", name=data.source.name)
        if data.source.download:
            source_type.download(config=data.source)
        raw_files = [Path(path) for path in source_type.raw_files(config=data.source)]
        partition_type = get(category="data/partitions", name=data.partition.name)
        # batch_size, download and seasonality live on the source config but do not
        # change the data.
        source_config = {
            key: value
            for key, value in asdict(data.source).items()
            if key not in {"batch_size", "download", "seasonality"}
        }
        data_key = _key(
            payload={
                "format": FORMAT_VERSION,
                "source": source_config,
                "partition": asdict(data.partition),
                "code": code(types=(source_type, partition_type, Parquet)),
                "raw": [
                    (str(path), path.stat().st_size, path.stat().st_mtime_ns)
                    for path in raw_files
                ],
            }
        )
        data_dir = (
            Path(self.config.root)
            / family(source_type=source_type)
            / "prepared"
            / data_key
        )
        if not (data_dir / "manifest.json").is_file():
            self._prepare_clients(
                experiment=experiment, raw_files=raw_files, data_dir=data_dir
            )
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        split_key = _key(
            payload={
                "data": data_key,
                "split": asdict(data.split),
                "scaler": asdict(data.scaler),
                "training": None
                if experiment.training is None
                else asdict(experiment.training),
                "training_iterations": experiment.strategy.iterations
                if experiment.training is not None
                else None,
                "training_code": code(
                    types=(get(category="training", name=experiment.training.name),)
                )
                if experiment.training is not None
                else None,
                "code": code(
                    types=(
                        get(category="data/splits", name=data.split.name),
                        get(category="scalers", name=data.scaler.name),
                    )
                ),
            }
        )
        split_dir = data_dir / "splits" / split_key
        if not (split_dir / "summary.json").is_file():
            self._prepare_splits(
                experiment=experiment,
                manifest=manifest,
                data_dir=data_dir,
                split_dir=split_dir,
            )
        summary = json.loads((split_dir / "summary.json").read_text(encoding="utf-8"))
        channels = tuple(manifest["channel_order"])
        input_length, _, offset_length = experiment.lengths()
        target_offset = input_length + offset_length
        records = [
            ClientRecord(
                client_id=entry["client_id"],
                name=entry["name"],
                store=self.config.name,
                values_path=str(data_dir / "clients" / f"{entry['client_id']}.parquet"),
                windows={
                    name: tuple((first, last) for first, last in entry["windows"][name])
                    for name in SPLITS
                },
                target_offset=target_offset,
                channels=channels,
                mean=tuple(entry["scaler"]["mean"]),
                scale=tuple(entry["scaler"]["scale"]),
                calendar=tuple(getattr(experiment.task, "calendar", ())),
            )
            for entry in summary["clients"]
        ]
        run_manifest = {
            "store": {
                "data_dir": str(data_dir),
                "split_dir": str(split_dir),
                "data_key": data_key,
                "split_key": split_key,
            },
            **{key: value for key, value in manifest.items() if key != "clients"},
            "clients": summary["clients"],
            "exclusions": summary["exclusions"],
        }
        return run_manifest, records

    @staticmethod
    def records(*, manifest: dict[str, Any], experiment: Any) -> list[ClientRecord]:
        """The client records of a saved run, read from its manifest: the data that
        run prepared, whatever the raw files hold now."""
        stores = manifest.get("groups") or [manifest]
        records = []
        for entry in manifest["clients"]:
            group = entry.get("group", 0)
            member = experiment.member(group=group)
            input_length, _, offset_length = member.lengths()
            data_dir = Path(stores[group]["store"]["data_dir"])
            file_id = entry.get("partition_client_id", entry["client_id"])
            records.append(
                ClientRecord(
                    client_id=entry["client_id"],
                    name=entry["name"],
                    store="Parquet",
                    values_path=str(data_dir / "clients" / f"{file_id}.parquet"),
                    windows={
                        name: tuple(
                            (first, last) for first, last in entry["windows"][name]
                        )
                        for name in SPLITS
                    },
                    target_offset=input_length + offset_length,
                    channels=tuple(stores[group]["channel_order"]),
                    mean=tuple(entry["scaler"]["mean"]),
                    scale=tuple(entry["scaler"]["scale"]),
                    calendar=tuple(getattr(member.task, "calendar", ())),
                    group=group,
                )
            )
        return records

    @staticmethod
    def _prepare_clients(
        *, experiment: Any, raw_files: list[Path], data_dir: Path
    ) -> None:
        data = experiment.data
        source_type = get(category="data/sources", name=data.source.name)
        source = source_type.load(config=data.source)
        partition_type = get(category="data/partitions", name=data.partition.name)
        clients = partition_type(config=data.partition).clients(source=source)
        build = data_dir.with_name(f".{data_dir.name}-{uuid.uuid4().hex[:8]}")
        (build / "clients").mkdir(parents=True)
        entries = []
        for client_id, name, frame in clients:
            columns = [
                pl.col(channel).cast(pl.Float32)
                for channel in data.source.input_channels
            ]
            if "sample" in frame.columns:
                frame = frame.sort("sample", "step")
                keys = ["sample", "step", "fold", "label"]
            else:
                # Series may carry official folds and per-step labels (SMD).
                keys = ["date", *(key for key in ("fold", "label") if key in frame)]
            frame.select(*keys, *columns).write_parquet(
                build / "clients" / f"{client_id}.parquet", compression="zstd"
            )
            entries.append({"client_id": client_id, "name": name, "rows": frame.height})
        manifest = {
            "format": FORMAT_VERSION,
            "raw_files": [
                {"path": str(path), "sha256": _sha256(path=path)} for path in raw_files
            ],
            "channel_order": list(data.source.input_channels),
            **{
                key: value
                for key, value in source.manifest_fields.items()
                if key != "raw_files"
            },
            "clients": entries,
        }
        (build / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
        )
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        _publish(build=build, target=data_dir)

    @staticmethod
    def _prepare_splits(
        *, experiment: Any, manifest: dict[str, Any], data_dir: Path, split_dir: Path
    ) -> None:
        data = experiment.data
        input_length, output_length, offset_length = experiment.lengths()
        channels = tuple(manifest["channel_order"])
        split_type = get(category="data/splits", name=data.split.name)
        split_policy = split_type(config=data.split)
        scaler_type = get(category="scalers", name=data.scaler.name)
        frames = {
            entry["client_id"]: pl.read_parquet(
                data_dir / "clients" / f"{entry['client_id']}.parquet"
            )
            for entry in manifest["clients"]
        }
        cuts = None
        if getattr(data.split, "cut", None) == "shared_calendar":
            cuts = split_policy.shared_cut_dates(
                frames=list(frames.values()),
                input_length=data.split.input_length,
                output_length=data.split.output_length,
                offset_length=data.split.offset_length,
                ratios=data.split.ratios,
                channels=channels,
            )
        build = split_dir.with_name(f".{split_dir.name}-{uuid.uuid4().hex[:8]}")
        build.mkdir(parents=True)
        clients: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for entry in manifest["clients"]:
            client_id = entry["client_id"]
            frame = frames[client_id]
            values = torch.from_numpy(frame.select(channels).to_numpy(writable=True))
            if hasattr(scaler_type, "validate_values"):
                scaler_type.validate_values(
                    values=values,
                    config=data.scaler,
                    source=data.source.name,
                    client=entry["name"],
                    channels=channels,
                )
            if "sample" in frame.columns:
                client = Parquet._panel_client(
                    experiment=experiment,
                    entry=entry,
                    frame=frame,
                    split_policy=split_policy,
                    scaler_type=scaler_type,
                    channels=channels,
                    values=values,
                )
                (exclusions if "exclusion_reason" in client else clients).append(client)
                continue
            if "fold" in frame.columns:
                splits = split_policy.series(
                    frame=frame,
                    input_length=input_length,
                    output_length=output_length,
                    offset_length=offset_length,
                )
            else:
                splits = split_policy.split(
                    frame=frame,
                    input_length=data.split.input_length,
                    output_length=data.split.output_length,
                    offset_length=data.split.offset_length,
                    ratios=data.split.ratios,
                    channels=channels,
                    lookback_crosses_boundary=data.split.lookback_crosses_boundary,
                    min_windows=data.split.min_windows,
                    cut_dates=cuts,
                )
            budget = getattr(data.split, "train_windows", 0)
            if budget:
                splits["train"] = splits["train"][-budget:]
            counts = {name: len(splits[name]) for name in SPLITS}
            total = sum(counts.values())
            dates = frame.get_column("date").to_list()
            client: dict[str, Any] = {
                "client_id": client_id,
                "name": entry["name"],
                "window_counts": counts,
                "realized_ratios": {
                    name: count / total if total else 0.0
                    for name, count in counts.items()
                },
                "cut_timestamps": {
                    name: dates[splits[name][0][1]] if splits[name] else None
                    for name in ("validation", "test")
                },
            }
            # A split with no validation share (ratio, ratio or window count 0)
            # does not require validation windows.
            validation_share = getattr(
                data.split,
                "validation_ratio",
                getattr(data.split, "validation_windows", 1),
            )
            too_few = [
                name
                for name, ratio in zip(
                    SPLITS,
                    getattr(data.split, "ratios", (1, validation_share, 1)),
                    strict=True,
                )
                if ratio > 0 and counts[name] < data.split.min_windows
            ]
            if too_few or not splits["train"]:
                client["exclusion_reason"] = (
                    f"fewer than min_windows={data.split.min_windows} "
                    f"in {too_few or ['train']}"
                )
                exclusions.append(client)
                continue
            fitting = splits["train"]
            train_start = 0
            if experiment.training is not None and info(
                category="training", name=experiment.training.name
            ).get("CONTINUAL", False):
                # The scaler sees only the first task's training rows.
                fitting = get(category="training", name=experiment.training.name)(
                    config=experiment.training
                ).released(
                    windows=fitting,
                    current_iteration=0,
                    iterations=experiment.strategy.iterations,
                    num_classes=None,
                )
                if not fitting:
                    raise ValueError("the first task has no training windows")
                train_start = min(start for start, _ in fitting)
            train_end = max(target + output_length for _, target in fitting)
            scaler = scaler_type.fit(
                values=values[train_start:train_end], config=data.scaler
            )
            client["scaler"] = {
                "mean": scaler.mean.tolist(),
                "scale": scaler.scale.tolist(),
            }
            offset = input_length + offset_length
            if any(
                target - start != offset
                for name in SPLITS
                for start, target in splits[name]
            ):
                raise ValueError(
                    "window targets must sit input_length + offset_length after starts"
                )
            client["windows"] = {
                name: ranges(starts=[start for start, _ in splits[name]])
                for name in SPLITS
            }
            clients.append(client)
        (build / "summary.json").write_text(
            json.dumps(
                {"clients": clients, "exclusions": exclusions},
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        split_dir.parent.mkdir(parents=True, exist_ok=True)
        _publish(build=build, target=split_dir)

    @staticmethod
    def _panel_client(
        *,
        experiment: Any,
        entry: dict[str, Any],
        frame: pl.DataFrame,
        split_policy: Any,
        scaler_type: Any,
        channels: tuple[str, ...],
        values: torch.Tensor,
    ) -> dict[str, Any]:
        """Split one client's samples; scale on its train samples' rows."""
        data = experiment.data
        samples = frame.unique(subset="sample", keep="first", maintain_order=True)
        splits = split_policy.split(samples=samples)
        counts = {name: len(splits[name]) for name in SPLITS}
        total = sum(counts.values())
        client: dict[str, Any] = {
            "client_id": entry["client_id"],
            "name": entry["name"],
            "window_counts": counts,
            "realized_ratios": {
                name: count / total if total else 0.0 for name, count in counts.items()
            },
        }
        too_few = [
            name
            for name in SPLITS
            if (name != "validation" or data.split.validation_ratio > 0)
            and counts[name] < data.split.min_windows
        ]
        if too_few:
            client["exclusion_reason"] = (
                f"fewer than min_windows={data.split.min_windows} in {too_few}"
            )
            return client
        values = values.reshape(samples.height, -1, len(channels))
        train_samples = sorted(splits["train"])
        if experiment.training is not None and info(
            category="training", name=experiment.training.name
        ).get("CONTINUAL", False):
            regime_type = get(category="training", name=experiment.training.name)
            regime = regime_type(config=experiment.training)
            labels = samples.get_column("label").to_list()
            for period in range(experiment.training.periods):
                selected = regime.windows(
                    windows=[(index, labels[index]) for index in train_samples],
                    period=period,
                    num_classes=data.source.num_classes,
                )
                if len(selected) < experiment.strategy.iterations:
                    raise ValueError(
                        "each continual task needs at least one example per round"
                    )
            train_samples = [
                index
                for index, _ in regime.released(
                    windows=[(index, labels[index]) for index in train_samples],
                    current_iteration=0,
                    iterations=experiment.strategy.iterations,
                    num_classes=data.source.num_classes,
                )
            ]
        scaler = scaler_type.fit(
            values=values[train_samples].reshape(-1, len(channels)), config=data.scaler
        )
        client["scaler"] = {
            "mean": scaler.mean.tolist(),
            "scale": scaler.scale.tolist(),
        }
        client["windows"] = {
            name: ranges(starts=sorted(splits[name])) for name in SPLITS
        }
        return client

    @staticmethod
    def load(*, record: ClientRecord) -> tuple[torch.Tensor, dict[str, list[Any]]]:
        """Read one client's scaled values and per-split windows.

        Series: values [steps, channels] (plus a last, unscaled label column when the
        source has per-step labels), windows (start, target). Panel: values
        [samples, steps, channels], windows (sample, label).
        """
        if "sample" in pl.read_parquet_schema(record.values_path):
            frame = pl.read_parquet(
                record.values_path, columns=["sample", "label", *record.channels]
            )
            samples = frame.unique(subset="sample", keep="first", maintain_order=True)
            values = torch.from_numpy(
                frame.select(record.channels).to_numpy(writable=True)
            ).reshape(samples.height, -1, len(record.channels))
            labels = samples.get_column("label").to_list()
            indices = None
            remap: dict[int, int] = {}
            if record.permissions is not None:
                indices = sorted(
                    {
                        index
                        for phase in record.permissions
                        for first, last in record.windows.get(phase, ())
                        for index in range(first, last + 1)
                    }
                )
                remap = {sample: index for index, sample in enumerate(indices)}
                values = values[indices]
            mean = torch.tensor(record.mean, dtype=values.dtype)
            scale = torch.tensor(record.scale, dtype=values.dtype)
            windows = {
                name: [
                    (remap[index] if indices is not None else index, labels[index])
                    for first, last in record.windows.get(name, ())
                    for index in range(first, last + 1)
                    if indices is None or index in remap
                ]
                for name in SPLITS
            }
            return (values - mean) / scale, windows
        labeled = "label" in pl.read_parquet_schema(record.values_path)
        frame = pl.read_parquet(
            record.values_path,
            columns=[
                *(["date"] if record.calendar else []),
                *record.channels,
                *(["label"] if labeled else []),
            ],
        )
        values = torch.from_numpy(frame.select(record.channels).to_numpy(writable=True))
        mean = torch.tensor(record.mean, dtype=values.dtype)
        scale = torch.tensor(record.scale, dtype=values.dtype)
        splits = {
            name: [
                (start, start + record.target_offset)
                for first, last in record.windows.get(name, ())
                for start in range(first, last + 1)
            ]
            for name in SPLITS
        }
        values = (values - mean) / scale
        if record.calendar:
            marks = features(dates=frame.get_column("date"), names=record.calendar)
            values = torch.cat([values, marks.to(values.dtype)], dim=1)
        if labeled:
            # Per-step labels ride along, unscaled, as the last column.
            labels = torch.tensor(frame.get_column("label").to_numpy())
            values = torch.cat([values, labels.to(values.dtype)[:, None]], dim=1)
        if record.visible_until is not None:
            values = values[: record.visible_until]
        return values, splits
