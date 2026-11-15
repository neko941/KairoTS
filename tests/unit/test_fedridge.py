from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from kairots import Experiment, run
from kairots.data.views.Offline import Offline
from kairots.models.Linear import Linear
from kairots.runtime.contracts import ClientState, Job
from kairots.strategies.FedRidge import FedRidge


def _worker(*, values: torch.Tensor, count: int, input_length: int) -> Any:
    experiment = SimpleNamespace(
        lengths=lambda: (input_length, 2, 0),
        data=SimpleNamespace(source=SimpleNamespace(batch_size=2)),
    )
    data = SimpleNamespace(
        values=values,
        splits={"train": [(index, index + input_length) for index in range(count)]},
    )
    return SimpleNamespace(
        client_experiment=lambda **_: experiment,
        client_data=lambda **_: data,
        components=lambda **_: (None, Offline(config=Offline.Config()), None),
    )


@pytest.mark.parametrize("input_length", [4, 1001])
def test_fedridge_matches_pooled_numpy_and_bills_statistics(input_length: int) -> None:
    strategy = FedRidge(config=FedRidge.Config())
    model = Linear(
        config=Linear.Config(
            input_length=input_length, output_length=2, input_channels=2
        )
    )
    initial = model.state_dict()
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(input_length + 12, 2, generator=generator)
    results, features, targets = [], [], []
    last_job: Job | None = None
    last_worker: Any = None
    for client_id, count in enumerate((3, 5, 7)):
        worker = _worker(
            values=values + client_id / 8, count=count, input_length=input_length
        )
        downlink = strategy.build_downlink(global_params=initial, client_id=client_id)
        assert downlink.billed_bytes == 0
        job = Job(
            operation_id=(0, client_id, "fit"),
            operation="fit",
            downlink=downlink,
            state=ClientState(),
            seed=941,
            client_id=client_id,
        )
        result = strategy.client_update(worker=worker, job=job)
        last_job, last_worker = job, worker
        assert result.state.optimizer_state == result.state.scheduler_state == {}
        assert result.state.version == 1
        results.append(result)
        client_data = worker.client_data(client_id=client_id)
        train_end = max(target + 2 for _, target in client_data.splits["train"])
        client_data.splits["test"] = [(count + 3, count + 3 + input_length)]
        client_data.values[train_end:] = 1e12
        poisoned = strategy.client_update(worker=worker, job=job)
        for key in result.uplink.tensors:
            torch.testing.assert_close(
                result.uplink.tensors[key], poisoned.uplink.tensors[key], rtol=0, atol=0
            )
        for batch_x, batch_y in worker.components(client_id=client_id)[1].batches(
            values=worker.client_data(client_id=client_id).values,
            windows=worker.client_data(client_id=client_id).splits["train"],
            input_length=input_length,
            output_length=2,
            offset_length=0,
            batch_size=2,
            shuffle=False,
            generator=generator,
        ):
            features.append(batch_x.transpose(1, 2).reshape(-1, input_length))
            targets.append(batch_y.transpose(1, 2).reshape(-1, 2))
    pooled_features = torch.cat(features)
    pooled_targets = torch.cat(targets).numpy()
    projection = strategy.projection(input_length=input_length)
    if projection is not None:
        pooled_features = pooled_features @ projection
    matrix = pooled_features.double().numpy()
    pooled_targets = pooled_targets.astype(np.float64)
    expected = np.linalg.solve(
        matrix.T @ matrix + 0.1 * np.eye(matrix.shape[1]), matrix.T @ pooled_targets
    )
    if projection is not None:
        expected = projection.double().numpy() @ expected
    fitted = strategy.aggregate(
        uplinks=[result.uplink for result in results], global_params=initial
    )
    np.testing.assert_allclose(
        # float32 statistics and solve against a float64 reference.
        fitted["forecasting.linear.weight"].double().numpy().T,
        expected,
        atol=1e-4,
        rtol=1e-4,
    )
    assert fitted["forecasting.linear.weight"].dtype == torch.float32
    assert not fitted["forecasting.linear.bias"].any()
    dimension = matrix.shape[1]
    for result in results:
        assert (
            result.uplink.billed_bytes
            == (dimension * (dimension + 1) // 2 + dimension * 2) * 4
        )
    messages = strategy.broadcast(
        uplinks=[result.uplink for result in results], global_params=fitted
    )
    assert [message.receiver for message in messages] == [
        result.uplink.sender for result in results
    ]
    assert all(message.billed_bytes == input_length * 2 * 4 for message in messages)
    with pytest.raises(ValueError, match="first fitting round"):
        assert last_job is not None
        last_job.operation_id = (1, 0, "fit")
        strategy.client_update(worker=last_worker, job=last_job)
    results[0].uplink.tensors["gram"][0] = torch.nan
    with pytest.raises(ValueError, match="invalid FedRidge"):
        strategy.aggregate(
            uplinks=[result.uplink for result in results], global_params=initial
        )


@pytest.mark.parametrize("gamma", [0.0, -1.0, float("nan"), float("inf")])
def test_fedridge_rejects_invalid_regularization(gamma: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        FedRidge.Config(gamma=gamma)


def test_fedridge_requirements_autofill_and_reject_conflicts() -> None:
    experiment = Experiment.from_dict(values={"strategy": {"name": "FedRidge"}})
    assert experiment.model.name == "Linear"
    assert experiment.learner.name == "Stateless"
    assert experiment.training.name == "Offline"
    for change in [
        {"model": {"name": "DLinear"}},
        {"learner": {"name": "Gradient"}},
        {"task": {"name": "Classification"}},
        {"strategy": {"name": "FedRidge", "iterations": 2}},
    ]:
        with pytest.raises((ValueError, TypeError)):
            Experiment.from_dict(values={"strategy": {"name": "FedRidge"}, **change})


@pytest.mark.parametrize("executor", ["Serial", "Ray"])
def test_fedridge_run_accounts_for_post_fit_broadcast(
    tmp_path: Path, executor: str
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for client_id in range(2):
        rows = ["DATE,TMP,DEW,WND_ANGLE,WND_RATE,SLP"]
        rows += [
            f"2020-01-01T{index:02}:00:00,{index + client_id},{index % 3},2,3,4"
            for index in range(24)
        ]
        (raw / f"station{client_id}.csv").write_text("\n".join(rows))
    experiment = Experiment.from_dict(
        values={
            "strategy": {"name": "FedRidge"},
            "data": {
                "source": {
                    "name": "TinyWeather5K",
                    "raw_path": str(raw),
                    "batch_size": 3,
                },
                "split": {"input_length": 4, "output_length": 2, "min_windows": 1},
                "store": {"root": str(tmp_path / "prepared")},
            },
            "runtime": {
                "name": executor,
                "runs_dir": str(tmp_path / "runs"),
                "device": "cpu",
            },
        }
    )
    result = run(experiment=experiment)
    row = result.rounds[0]
    assert len(result.rounds) == 1
    assert row["downlink_bytes"] == 2 * 4 * 2 * 4
    assert row["uplink_bytes"] == 2 * (4 * 5 // 2 + 4 * 2) * 4
    assert np.isfinite(row["generalization_test_pooled_mse"])
