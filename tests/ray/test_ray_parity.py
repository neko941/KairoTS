from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.runtime.orchestrator import run


@pytest.mark.parametrize(
    "partition", ["Dirichlet", "ExtendedDirichlet", "Pathological", "ClassPerClient"]
)
def test_labeled_partition_serial_matches_ray(
    workspace_tmp_path: Path,
    partition: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from integration.test_classification import _config, _raw

    from kairots.config.schema import Experiment

    # Match Ray's one CPU per actor, including floating-point reduction order.
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    torch.set_num_threads(1)
    # Ray workers inherit explicit thread settings instead of their CPU allocation.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    raw = _raw(root=workspace_tmp_path / "raw")
    results = []
    for executor in ("Serial", "Ray"):
        values = _config(root=workspace_tmp_path / executor, raw=raw, iterations=1)
        values["data"]["partition"] = {
            "name": partition,
            "num_clients": 2,
        }
        if partition == "Dirichlet":
            values["data"]["partition"]["min_samples"] = 10
        values["runtime"].update(name=executor, save_checkpoints=True)
        result = run(experiment=Experiment.from_dict(values=values))
        manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text())
        checkpoint = torch.load(
            Path(result.run_dir) / "checkpoints/final.pt", weights_only=True
        )
        results.append((manifest["clients"], checkpoint))
    assert results[0][0] == results[1][0]
    for name, value in results[0][1].items():
        torch.testing.assert_close(value, results[1][1][name], rtol=1e-5, atol=1e-6)


def test_serial_ray_parity_on_two_clients(workspace_tmp_path: Path) -> None:
    tmp_path = workspace_tmp_path
    raw = tmp_path / "raw"
    write_ett_sources(path=raw)
    serial_experiment = ett_experiment(
        raw_path=raw,
        runs_dir=tmp_path / "serial",
        executor="Serial",
        save_checkpoints=True,
    )
    ray_experiment = ett_experiment(
        raw_path=raw,
        runs_dir=tmp_path / "ray",
        executor="Ray",
        save_checkpoints=True,
    )
    serial = run(experiment=serial_experiment)
    distributed = run(experiment=ray_experiment)
    for key in (
        "generalization_val_pooled_mse",
        "generalization_val_mean_client_mse",
        "generalization_val_pooled_mae",
        "generalization_val_mean_client_mae",
        "generalization_test_pooled_mse",
        "generalization_test_mean_client_mae",
    ):
        assert serial.rounds[0][key] == pytest.approx(
            distributed.rounds[0][key], rel=1e-5, abs=1e-6
        )


def test_local_only_serial_matches_ray(workspace_tmp_path: Path) -> None:
    tmp_path = workspace_tmp_path
    raw = tmp_path / "raw"
    write_ett_sources(path=raw)
    serial_experiment = ett_experiment(
        raw_path=raw,
        runs_dir=tmp_path / "serial",
        executor="Serial",
        strategy="LocalOnly",
        iterations=2,
        save_checkpoints=True,
    )
    ray_experiment = ett_experiment(
        raw_path=raw,
        runs_dir=tmp_path / "ray",
        executor="Ray",
        strategy="LocalOnly",
        iterations=2,
        save_checkpoints=True,
    )
    serial = run(experiment=serial_experiment)
    distributed = run(experiment=ray_experiment)
    assert serial.rounds[-1]["generalization_test_pooled_mse"] == pytest.approx(
        distributed.rounds[-1]["generalization_test_pooled_mse"], rel=1e-6
    )


def test_ray_reuses_one_worker_for_multiple_clients_without_state_leakage(
    workspace_tmp_path: Path,
) -> None:
    import json

    from kairots.config.schema import Experiment

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    client_rows = []
    for workers in (1, 2):
        values = ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / f"ray-{workers}-workers",
            strategy="LocalOnly",
            iterations=3,
            executor="Ray",
            save_checkpoints=True,
        ).to_dict()
        values["runtime"]["num_workers"] = workers
        result = run(experiment=Experiment.from_dict(values=values))
        run_dir = Path(result.run_dir)
        rounds = [
            json.loads(line)
            for line in (run_dir / "rounds.jsonl").read_text().splitlines()
        ]
        assert [row["selected_clients"] for row in rounds] == [[0, 1]] * 3
        client_rows.append(
            [
                json.loads(line)
                for line in (run_dir / "clients.jsonl").read_text().splitlines()
            ]
        )

    assert [
        {key: value for key, value in row.items() if key != "timing"}
        for row in client_rows[0]
    ] == [
        {key: value for key, value in row.items() if key != "timing"}
        for row in client_rows[1]
    ]


@pytest.mark.parametrize(
    ("model_name", "learner_name"),
    [("Naive", "Stateless"), ("XGBoost", "Estimator")],
)
def test_stateless_and_estimator_methods_match_between_serial_and_ray(
    workspace_tmp_path: Path, model_name: str, learner_name: str
) -> None:
    from kairots.config.schema import Experiment

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    results = []
    for executor in ("Serial", "Ray"):
        values = ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / f"{model_name}-{executor}",
            strategy="LocalOnly",
            iterations=2,
            executor=executor,
        ).to_dict()
        if model_name == "XGBoost":
            values["model"] = {
                "name": model_name,
                "num_trees": 3,
                "max_depth": 2,
                "shrinkage": 0.1,
            }
        else:
            values["model"] = {"name": model_name}
        values["learner"] = {"name": learner_name}
        experiment = Experiment.from_dict(values=values)
        results.append(run(experiment=experiment))

    serial, distributed = results
    assert len(serial.rounds) == len(distributed.rounds) == 2
    for serial_row, ray_row in zip(serial.rounds, distributed.rounds, strict=True):
        for key, value in serial_row.items():
            if not key.endswith("_seconds") and key != "selected_clients":
                assert ray_row[key] == pytest.approx(value, rel=1e-6, abs=1e-7)


def test_ray_completion_order_does_not_change_client_results(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from kairots.config.schema import Experiment
    from kairots.runtime.contracts import Job, Result
    from kairots.runtime.worker import Worker

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    baseline_values = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "serial",
        strategy="LocalOnly",
        iterations=2,
        executor="Serial",
    ).to_dict()
    baseline = run(experiment=Experiment.from_dict(values=baseline_values))
    baseline_rows = [
        json.loads(line)
        for line in (Path(baseline.run_dir) / "clients.jsonl").read_text().splitlines()
    ]

    events = workspace_tmp_path / "completion-order.jsonl"
    original_execute = Worker.execute

    def delayed_execute(
        self: Worker, *, job: Job, broadcast: dict[str, torch.Tensor] | None = None
    ) -> Result:
        descriptor = os.open(events, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(
                descriptor,
                f"start {job.operation_id[0]} {job.client_id}\n".encode(),
            )
        finally:
            os.close(descriptor)
        if job.client_id == 0:
            time.sleep(0.5)
        result = original_execute(self, job=job, broadcast=broadcast)
        descriptor = os.open(events, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(
                descriptor,
                f"end {job.operation_id[0]} {job.client_id}\n".encode(),
            )
        finally:
            os.close(descriptor)
        return result

    monkeypatch.setattr(Worker, "execute", delayed_execute)
    ray_values = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "ray",
        strategy="LocalOnly",
        iterations=2,
        executor="Ray",
    ).to_dict()
    ray_values["runtime"]["num_workers"] = 2
    distributed = run(experiment=Experiment.from_dict(values=ray_values))
    distributed_rows = [
        json.loads(line)
        for line in (Path(distributed.run_dir) / "clients.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [
        {key: value for key, value in row.items() if key != "timing"}
        for row in distributed_rows
    ] == [
        {key: value for key, value in row.items() if key != "timing"}
        for row in baseline_rows
    ]

    completed = {}
    for index, line in enumerate(events.read_text().splitlines()):
        phase, iteration, client = line.split()
        if phase == "end":
            completed[(int(iteration), int(client))] = index
    assert all(
        completed[(iteration, 1)] < completed[(iteration, 0)] for iteration in range(2)
    )


def test_online_replay_replay_serial_matches_ray(workspace_tmp_path: Path) -> None:
    from kairots.config.schema import Experiment

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    results = []
    for executor in ("Serial", "Ray"):
        values = ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / executor,
            executor=executor,
            iterations=2,
        ).to_dict()
        values["data"]["view"] = {"name": "Ordered"}
        values["evaluation"] = {
            "name": "Prequential",
            "regime": "Online",
            "replay_size": 4,
            "capacity": 8,
        }
        results.append(run(experiment=Experiment.from_dict(values=values)))
    serial, distributed = results
    assert "generalization_test_forgetting_pooled_mse" in serial.rounds[-1]
    for key, value in serial.rounds[-1].items():
        if not key.endswith("_seconds"):
            assert distributed.rounds[-1][key] == pytest.approx(
                value, rel=1e-5, abs=1e-6
            )


def test_federated_reinforce_serial_matches_ray(workspace_tmp_path: Path) -> None:
    from kairots.config.schema import Experiment

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    results = []
    for executor in ("Serial", "Ray"):
        values = ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / executor,
            executor=executor,
            iterations=2,
        ).to_dict()
        values["task"] = {"name": "Control"}
        values["data"]["source"]["input_channels"] = ["HUFL"]
        values["data"]["scaler"] = {"name": "Identity"}
        values["model"] = {"name": "Policy", "hidden": 8}
        values["learner"] = {"name": "REINFORCE", "learning_rate": 0.01}
        values.pop("evaluation")
        results.append(run(experiment=Experiment.from_dict(values=values)))
    serial, distributed = results
    assert serial.rounds[-1]["generalization_test_pooled_return"] == pytest.approx(
        distributed.rounds[-1]["generalization_test_pooled_return"], rel=1e-5, abs=1e-6
    )
    assert len(serial.rounds[-1]["selected_clients"]) == 2


def test_scaffold_serial_matches_ray(workspace_tmp_path: Path) -> None:
    from kairots.config.schema import Experiment

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    results = []
    for executor in ("Serial", "Ray"):
        values = ett_experiment(
            raw_path=raw,
            runs_dir=workspace_tmp_path / executor,
            executor=executor,
            strategy="SCAFFOLD",
            optimizer="SGD",
            iterations=2,
        ).to_dict()
        results.append(run(experiment=Experiment.from_dict(values=values)))
    serial, distributed = results
    assert serial.rounds[-1]["generalization_test_pooled_mse"] == pytest.approx(
        distributed.rounds[-1]["generalization_test_pooled_mse"], rel=1e-5, abs=1e-6
    )


def test_ray_puts_client_metadata_once_and_shares_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from kairots.data.contracts import ClientRecord

    ray_module = importlib.import_module("kairots.runtime.executors.Ray")

    put_values: list[ClientRecord] = []
    delivered: list[tuple[int, object]] = []

    class ActorMethod:
        def remote(self, *, client_id: int, data: object) -> None:
            delivered.append((client_id, data))

    class Actor:
        cache_client = ActorMethod()

    references: list[object] = []

    def put(value: ClientRecord) -> object:
        put_values.append(value)
        reference = object()
        references.append(reference)
        return reference

    monkeypatch.setattr(ray_module.ray, "put", put)
    executor = ray_module.Ray.__new__(ray_module.Ray)
    executor.actors = [Actor(), Actor()]
    executor.owners = {}

    for client_id in range(64):
        executor.cache_client(
            client_id=client_id,
            data=ClientRecord(
                client_id=client_id,
                name=str(client_id),
                store="Parquet",
                values_path=f"{client_id}.parquet",
                windows={},
                target_offset=0,
                channels=("OT",),
                mean=(0.0,),
                scale=(1.0,),
            ),
        )

    assert len(put_values) == 64
    assert len(delivered) == 128
    for client_id, reference in enumerate(references):
        assert delivered[2 * client_id : 2 * client_id + 2] == [
            (client_id, reference),
            (client_id, reference),
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_ray_reserved_gpu_runs_and_scores_a_model_on_cuda(
    workspace_tmp_path: Path,
) -> None:
    from kairots.config.schema import Experiment

    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    experiment = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        executor="Ray",
        iterations=1,
    )
    values = experiment.to_dict()
    values["runtime"].update(
        device="cuda", num_gpus=0.1, num_workers=1, save_checkpoints=False
    )
    result = run(experiment=Experiment.from_dict(values=values))
    assert torch.isfinite(
        torch.tensor(result.rounds[-1]["generalization_test_pooled_mse"])
    )
