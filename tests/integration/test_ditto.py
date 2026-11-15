from __future__ import annotations

from pathlib import Path

import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.runtime.orchestrator import run


def test_ditto_personal_models_differ_from_global(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    result = run(
        experiment=ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / "runs",
            strategy="Ditto",
            iterations=2,
            save_checkpoints=True,
        )
    )
    final_path = Path(result.run_dir) / "checkpoints" / "final.pt"
    final_clients_path = Path(result.run_dir) / "checkpoints" / "final_clients.pt"
    assert final_clients_path.exists()
    global_model = torch.load(final_path, weights_only=True)
    client_models = torch.load(final_clients_path, weights_only=True)
    assert len(client_models) == 2
    assert any(
        not torch.equal(client_models[0][key], global_model[key])
        for key in global_model
    )
