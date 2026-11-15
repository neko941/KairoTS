from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from kairots.config.schema import Experiment
from kairots.runtime.orchestrator import run


def test_pems08_run_prepares_graph_and_three_clients(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw" / "PEMS08"
    raw.mkdir(parents=True)
    values = np.arange(400 * 3 * 3, dtype=np.float32).reshape(400, 3, 3)
    np.savez(raw / "PEMS08.npz", data=values)
    with (raw / "PEMS08.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("from", "to", "cost"))
        writer.writerows(((0, 1, 1.0), (1, 2, 2.0)))
    config = json.loads(
        (Path(__file__).parents[2] / "configs/forecasting/PeMS08.json").read_text(
            encoding="utf-8"
        )
    )
    config["data"]["source"].update(
        raw_path=str(workspace_tmp_path / "raw"), download=False
    )
    config["data"]["partition"]["nodes"] = [0, 1, 2]
    config["data"]["split"].update(input_length=16, output_length=4)
    config["data"]["store"]["root"] = str(workspace_tmp_path / "datasets")
    config["strategy"]["iterations"] = 1
    config["runtime"].update(
        device="cpu", runs_dir=str(workspace_tmp_path / "runs"), save_checkpoints=False
    )
    result = run(experiment=Experiment.from_dict(values=config))
    assert math.isfinite(result.rounds[-1]["generalization_test_pooled_mse"])
    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text())
    assert len(manifest["clients"]) == 3
    assert len(manifest["graph"]) == 2
    assert manifest["num_nodes"] == 3
