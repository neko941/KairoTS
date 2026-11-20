from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from kairots.config.schema import Experiment
from kairots.runtime.contracts import ClientState, Message, Result
from kairots.runtime.state import StateStore
from kairots.runtime.worker import Batches, _sample_windows


def test_batches_limit_replays_stream_and_counts_consumed_samples() -> None:
    def stream(
        *, generator: torch.Generator
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(4):
            yield torch.rand(3, generator=generator), torch.zeros(3)

    batches = Batches(
        stream=stream,
        state=torch.Generator().manual_seed(7).get_state(),
        length=2,
    )
    first = list(batches)
    assert len(batches) == 2 and batches.sample_count == 6
    assert torch.equal(first[0][0], list(batches)[0][0])
    assert batches.sample_count == 6


def test_batches_per_epoch_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batches_per_epoch must be positive"):
        Experiment.from_dict(values={"learner": {"batches_per_epoch": 0}})


def test_zero_shot_model_config_compares_json_normalized_values() -> None:
    from dataclasses import asdict

    from kairots.models.TimeGrad import TimeGrad
    from kairots.runtime.orchestrator import _json_default, _same_model_config

    target = TimeGrad.Config()
    source = json.loads(json.dumps(asdict(target), default=_json_default))

    assert isinstance(target.lags, tuple)
    assert isinstance(source["lags"], list)
    assert _same_model_config(source=source, target=target)
    source["conditioning"] = -1
    assert not _same_model_config(source=source, target=target)


def test_sampled_training_windows_use_replacement() -> None:
    windows = [(0, 1), (1, 2)]
    sampled = _sample_windows(
        windows=windows,
        batch_size=2,
        batches_per_epoch=3,
        generator=torch.Generator().manual_seed(5),
    )
    assert len(sampled) == 6
    assert set(sampled) == set(windows)
    assert len(set(sampled)) < len(sampled)


def test_identity_codec_bills_tensor_bytes_plus_scalars() -> None:
    message = Message(
        sender="server",
        receiver="client",
        tensors={
            "a": torch.zeros(3, dtype=torch.float32),
            "b": torch.zeros(2, dtype=torch.float64),
        },
        scalars={"step": 1.0, "ratio": 0.5},
    )
    assert message.billed_bytes == 3 * 4 + 2 * 8 + 2 * 4


def test_state_store_rejects_duplicate_operation() -> None:
    store = StateStore(num_clients=1)
    result = Result(
        operation_id=(0, 0, "fit"),
        uplink=Message(sender="client", receiver="server"),
        state=ClientState(version=1),
        metrics={},
        train_samples=1,
    )
    store.accept(result=result)
    with pytest.raises(ValueError, match="duplicate operation_id"):
        store.accept(result=result)


def test_same_seed_same_run_and_new_seed_differs(tmp_path: Path) -> None:

    from kairots import Experiment, run

    config = json.loads(
        (
            Path(__file__).parents[2]
            / "configs/examples/forecasting/ETDatasetHour.json"
        ).read_text(encoding="utf-8")
    )
    raw = Path(__file__).parents[2] / config["data"]["source"]["raw_path"]
    if not raw.exists():
        pytest.skip("ETT raw data not downloaded; run `python -m kairots prepare`")
    config["data"]["source"].update(raw_path=str(raw), download=False)
    config["strategy"].update(iterations=1, join_ratio=0.5)
    config["runtime"].update(runs_dir=str(tmp_path), save_checkpoints=False)
    config["data"]["store"]["root"] = str(tmp_path / "datasets")
    config["model"] = {"name": "DLinear"}

    def final(*, seed: int) -> tuple[list[int], float]:
        config["runtime"]["seed"] = seed
        result = run(experiment=Experiment.from_dict(values=config))
        last = result.rounds[-1]
        return last["selected_clients"], last["generalization_test_pooled_mse"]

    assert final(seed=1) == final(seed=1)
    assert final(seed=1) != final(seed=2)


def test_self_addressed_message_bills_nothing() -> None:
    message = Message(sender="server", receiver="server", tensors={"p": torch.ones(3)})
    assert message.billed_bytes == 0


def test_worker_resolves_each_groups_task_from_its_channels() -> None:
    from types import SimpleNamespace

    from kairots.config.schema import Experiment
    from kairots.runtime.worker import Worker

    experiment = Experiment.from_dict(
        values={
            "data": {"source": {"input_channels": ["HUFL", "OT"]}},
            "task": {"target_channels": ["OT"]},
            "model": {"name": "TiDE"},
            "groups": [{}, {"data": {"source": {"input_channels": ["OT", "HUFL"]}}}],
        }
    )
    worker = Worker(
        experiment=experiment,
        strategy=None,
        task=None,
        learner_type=None,
        loss_function=None,
        view=None,
    )
    for client_id in (0, 1):
        worker.records[client_id] = SimpleNamespace(group=client_id)  # type: ignore[assignment]
    first, second = (worker.components(client_id=client_id)[0] for client_id in (0, 1))
    assert (first.target_index, second.target_index) == ([1], [0])


def test_worker_client_data_cache_stays_bounded_for_realistic_series(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fixtures.synthetic import ett_experiment

    from kairots.data.contracts import ClientRecord
    from kairots.runtime import worker as worker_module
    from kairots.runtime.worker import Worker

    worker = Worker(
        experiment=ett_experiment(
            raw_path=workspace_tmp_path / "raw", runs_dir=workspace_tmp_path / "runs"
        ),
        strategy=None,
        task=None,
        learner_type=None,
        loss_function=None,
        view=None,
    )
    loaded: list[int] = []

    class Store:
        @staticmethod
        def load(
            *, record: ClientRecord
        ) -> tuple[torch.Tensor, dict[str, list[tuple[int, int]]]]:
            loaded.append(record.client_id)
            return torch.zeros((17_420, 7)), {"train": []}

    monkeypatch.setattr(worker_module, "get", lambda **_: Store)
    for client_id in range(40):
        worker.cache_client(
            client_id=client_id,
            data=ClientRecord(
                client_id=client_id,
                name=str(client_id),
                store="Parquet",
                values_path="",
                windows={},
                target_offset=0,
                channels=(),
                mean=(),
                scale=(),
            ),
        )
        worker.client_data(client_id=client_id)

    assert len(worker.clients) == Worker.CLIENT_DATA_CACHE_LIMIT == 8
    cached_bytes = sum(
        data.values.numel() * data.values.element_size()
        for data in worker.clients.values()
    )
    assert cached_bytes <= 8 * 17_420 * 7 * 4
    worker.client_data(client_id=32)
    worker.client_data(client_id=31)
    assert 32 in worker.clients and 33 not in worker.clients
    worker.client_data(client_id=33)
    assert len(worker.clients) == 8
    assert loaded.count(32) == 1
    assert loaded.count(31) == 2
    assert loaded.count(33) == 2


@pytest.mark.parametrize(
    ("policy", "remote", "num_gpus", "cuda_available", "expected"),
    [
        ("cpu", True, 0.0, True, "cpu"),
        ("auto", True, 0.0, True, "cpu"),
        ("auto", True, 0.5, True, "cuda"),
        ("auto", False, 0.0, True, "cuda"),
        ("auto", False, 0.0, False, "cpu"),
    ],
)
def test_execution_device_honors_resource_and_device_policy(
    policy: str,
    remote: bool,
    num_gpus: float,
    cuda_available: bool,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kairots.runtime.device import execution_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    assert (
        str(execution_device(policy=policy, remote=remote, num_gpus=num_gpus))
        == expected
    )


def test_requested_cuda_without_visible_device_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kairots.runtime.device import execution_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        execution_device(policy="cuda", remote=True, num_gpus=0.5)
    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        execution_device(policy="auto", remote=True, num_gpus=0.5)
