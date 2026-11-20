from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kairots.config.schema import Experiment
from kairots.data.sources.M4.M4Hourly import M4Hourly
from kairots.runtime.orchestrator import run

LENGTHS = {"H1": 40, "H2": 52, "H3": 46}


def _write_csvs(*, raw: Path) -> None:
    longest = max(LENGTHS.values())
    for fold, size in (("Train", None), ("Test", 4)):
        width = size or longest
        lines = [",".join(f'"V{column}"' for column in range(1, width + 2))]
        for offset, (series, length) in enumerate(LENGTHS.items()):
            count = size or length
            values = [f"{100 + offset + step % 24}" for step in range(count)]
            lines.append(",".join([f'"{series}"', *values, *[""] * (width - count)]))
        (raw / fold).mkdir(parents=True)
        (raw / fold / f"Hourly-{fold.lower()}.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    (raw / "M4-info.csv").write_text(
        "M4id,category,Frequency,Horizon,SP,StartingDate\n"
        + "".join(f"{series},Macro,24,4,Hourly,01-01-00\n" for series in LENGTHS),
        encoding="utf-8",
    )


def test_m4_hourly_run_scores_mase_per_series(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(M4Hourly, "SERIES_COUNT", len(LENGTHS))
    monkeypatch.setattr(M4Hourly, "HORIZON", 4)
    raw = workspace_tmp_path / "raw"
    _write_csvs(raw=raw)
    config = json.loads(
        (
            Path(__file__).parents[2] / "configs/examples/forecasting/M4Hourly.json"
        ).read_text(encoding="utf-8")
    )
    config["evaluation"]["metrics"]["test"].remove("M4OWA")
    config["data"]["source"].update(raw_path=str(raw), download=False)
    config["data"]["split"].update(input_length=8, output_length=4, train_stride=4)
    config["data"]["store"]["root"] = str(workspace_tmp_path / "datasets")
    config["strategy"]["iterations"] = 1
    config["runtime"].update(
        device="cpu", runs_dir=str(workspace_tmp_path / "runs"), save_checkpoints=False
    )
    result = run(experiment=Experiment.from_dict(values=config))
    last = result.rounds[-1]
    assert math.isfinite(last["generalization_test_pooled_mase"])
    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text())
    assert [client["name"] for client in manifest["clients"]] == list(LENGTHS)
    assert all(client["window_counts"]["test"] == 1 for client in manifest["clients"])
