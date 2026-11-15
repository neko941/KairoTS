"""Tests for analysis reporting across runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kairots.__main__ import _report
from kairots.analysis.report import collect, markdown, report, summarize


def _setup_fake_runs(*, root: Path) -> None:
    runs = (
        ("first_run", "Forecasting", "ETTh1", "Linear", 1, "completed", 0.2),
        ("second_run", "Forecasting", "ETTh1", "Linear", 2, "completed", 0.4),
        ("failed_run", "Classification", "BasicMotions", "DLinear", 1, "failed", None),
    )
    for directory, task, source, model, seed, status, mse in runs:
        run_directory = root / directory
        run_directory.mkdir(parents=True)
        config = {
            "resolved": {
                "task": {"name": task},
                "data": {"source": {"name": source}},
                "model": {"name": model},
                "strategy": {"name": "FedAvg"},
                "runtime": {"seed": seed},
            },
            "seed": seed,
        }
        summary: dict[str, object] = {
            "status": status,
            "final_metrics": (
                {}
                if mse is None
                else {
                    "current_iter": 9,
                    "selected_clients": [0, 1],
                    "generalization_test_pooled_mse": mse,
                }
            ),
            "total_time_seconds": 1.0,
            "total_bytes": 0,
        }
        if seed == 2 and mse is not None:
            # The best validation round is reported, not the worse final round.
            summary["best_val_round"] = 3
            summary["best_val_metrics"] = summary["final_metrics"]
            summary["final_metrics"] = {"generalization_test_pooled_mse": 9.0}
        if mse is None:
            summary["error"] = "RuntimeError: division by zero"
        (run_directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run_directory / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )


def test_collect_summarize_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_fake_runs(root=tmp_path)

    rows = collect(root=tmp_path)
    assert len(rows) == 3
    assert not any(
        "current_iter" in item or "selected_clients" in item for item in rows
    )
    for expected_key in (
        "run",
        "task",
        "source",
        "model",
        "strategy",
        "config",
        "seed",
        "status",
        "error",
        "total_time_seconds",
        "total_bytes",
    ):
        assert all(expected_key in item for item in rows)

    summarized = summarize(rows=rows, metrics=["generalization_test_pooled_mse"])
    assert len(summarized) == 1
    group_entry = summarized[0]
    assert group_entry["runs"] == 2
    assert group_entry["generalization_test_pooled_mse_mean"] == pytest.approx(0.3)
    assert group_entry["generalization_test_pooled_mse_std"] == pytest.approx(0.1)

    text_report = report(root=tmp_path, metrics=["generalization_test_pooled_mse"])
    assert "0.3 +/- 0.1" in text_report
    assert "Failed runs: 1" in text_report

    _report(arguments=[str(tmp_path), "--metrics", "generalization_test_pooled_mse"])
    output = capsys.readouterr().out
    assert (
        "| task | source | model | strategy | config | runs | "
        "generalization_test_pooled_mse |" in output
    )
    assert "0.3 +/- 0.1" in output
    assert "Failed runs: 1" in output


def test_markdown_renders_table() -> None:
    rows = [{"task": "Forecasting", "score": 0.12345, "missing": None}]
    rendered = markdown(rows=rows, columns=["task", "score", "missing"])
    assert rendered == (
        "| task | score | missing |\n|---|---|---|\n| Forecasting | 0.1235 |  |"
    )
