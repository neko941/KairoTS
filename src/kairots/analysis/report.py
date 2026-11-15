"""One table across runs: read each run directory's config and summary, group runs
that share one resolved config apart from `runtime` (seed, device, executor, paths),
and report the mean and standard deviation of each selected metric over the completed
runs of a group (its seeds). A run that picked a best validation round reports
that round's scores. `config` is a short hash naming the group's settings.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def collect(*, root: Path) -> list[dict[str, Any]]:
    """One row per run directory under `root`: its identity, status and the numeric
    metrics (`current_iteration` excluded) of the best validation round when one was
    picked, else of the final round."""
    rows = []
    for summary_path in sorted(root.rglob("summary.json")):
        run_directory = summary_path.parent
        if not (run_directory / "config.json").is_file():
            continue
        config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        resolved = config["resolved"]
        metrics = (
            summary["best_val_metrics"]
            if summary.get("best_val_round") is not None
            else summary["final_metrics"]
        )
        settings = {key: value for key, value in resolved.items() if key != "runtime"}
        rows.append(
            {
                "run": run_directory.relative_to(root).as_posix(),
                "task": resolved["task"]["name"],
                "source": resolved["data"]["source"]["name"],
                "model": resolved["model"]["name"],
                "strategy": resolved["strategy"]["name"],
                "config": hashlib.sha256(
                    json.dumps(settings, sort_keys=True).encode("utf-8")
                ).hexdigest()[:8],
                "seed": config["seed"],
                "status": summary["status"],
                "error": summary.get("error", ""),
                **{
                    name: value
                    for name, value in metrics.items()
                    if name != "current_iter"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                },
                "total_time_seconds": summary["total_time_seconds"],
                "total_bytes": summary["total_bytes"],
            }
        )
    return rows


def summarize(
    *, rows: list[dict[str, Any]], metrics: list[str]
) -> list[dict[str, Any]]:
    """Completed runs grouped by task, source, model, strategy and config; each
    metric's mean and population standard deviation over the runs that report it
    (None if none)."""
    fields = ("task", "source", "model", "strategy", "config")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "completed":
            groups.setdefault(tuple(row[name] for name in fields), []).append(row)
    output = []
    for key, members in sorted(groups.items()):
        group: dict[str, Any] = {**dict(zip(fields, key)), "runs": len(members)}
        for metric in metrics:
            values = [
                row[metric]
                for row in members
                if isinstance(row.get(metric), (int, float))
                and not isinstance(row[metric], bool)
                and math.isfinite(row[metric])
            ]
            group[f"{metric}_n"] = len(values)
            group[f"{metric}_mean"] = statistics.mean(values) if values else None
            group[f"{metric}_std"] = statistics.pstdev(values) if values else None
        output.append(group)
    return output


def markdown(*, rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as a GitHub markdown table for the specified columns."""
    if not columns:
        return ""
    header = f"| {' | '.join(columns)} |"
    separator = f"|{'|'.join('---' for _ in columns)}|"
    lines = [header, separator]
    for row in rows:
        cells: list[str] = []
        for column in columns:
            value = row.get(column)
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def report(*, root: Path, metrics: list[str]) -> str:
    """Markdown table of `summarize`, one `mean +/- std` cell per metric, then the
    number of failed runs."""
    rows = collect(root=root)
    columns = ["task", "source", "model", "strategy", "config", "runs", *metrics]
    formatted = [
        {
            **group,
            **{
                metric: None
                if group[f"{metric}_mean"] is None
                else f"{group[f'{metric}_mean']:.4g} +/- {group[f'{metric}_std']:.2g}"
                for metric in metrics
            },
        }
        for group in summarize(rows=rows, metrics=metrics)
    ]
    failed_count = sum(row["status"] == "failed" for row in rows)
    table = markdown(rows=formatted, columns=columns)
    return f"{table}\n\nFailed runs: {failed_count}"
