from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from fixtures.synthetic import write_ett_sources
from torch import Tensor, nn

from kairots.config.schema import Experiment
from kairots.data.stores.Parquet import Parquet
from kairots.registry import get
from kairots.training.Online import Online


class _Mean(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return batch_x[:, -1:, :] + self.bias


def test_online_regime_trains_only_on_observed_targets() -> None:
    model = _Mean()
    regime = Online(config=Online.Config(steps=1))
    regime.start(optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    trained: list[int] = []

    def loss_function(*, prediction: Tensor, target: Tensor) -> Tensor:
        trained.append(len(target))
        return (prediction - target).pow(2).mean()

    batch_x = torch.ones(3, 4, 1)
    regime.observe(batch_x=batch_x, batch_y=torch.zeros(3, 1, 1), ends=[10, 11, 12])
    regime.release(model=model, observed=9, loss_function=loss_function)
    assert trained == [] and regime.updates == 0
    regime.release(model=model, observed=11, loss_function=loss_function)
    assert trained == [2] and [item[2] for item in regime.pending] == [12]
    regime.release(model=model, observed=11, loss_function=loss_function)
    assert trained == [2]


def _setup(*, workspace_tmp_path: Path, steps: int) -> tuple[Any, Any, dict[str, Any]]:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    experiment = Experiment.from_dict(
        values={
            "data": {
                "split": {
                    "input_length": 8,
                    "output_length": 3,
                    "ratios": [0.6, 0.2, 0.2],
                },
                "source": {
                    "name": "ETDatasetHour",
                    "raw_path": str(raw),
                    "download": False,
                    "batch_size": 4,
                },
                "view": {"name": "Ordered"},
                "store": {"root": str(workspace_tmp_path / "datasets")},
            },
            "learner": {"learning_rate": 0.01},
            "evaluation": {"name": "Prequential", "steps": steps},
            "runtime": {"device": "cpu", "runs_dir": str(workspace_tmp_path / "runs")},
        }
    )
    torch.manual_seed(0)
    model = get(category="models", name=experiment.model.name)(config=experiment.model)
    _, records = Parquet(config=experiment.data.store).prepare(experiment=experiment)
    loss_type = get(category="metrics", name=experiment.evaluation.loss)
    protocol = get(category="evaluation", name="Prequential")(
        config=experiment.evaluation, loss_function=loss_type(config=loss_type.Config())
    )
    arguments = {
        "phase": "test",
        "model": model,
        "clients": [Parquet.load(record=record) for record in records],
        "scales": [
            (torch.tensor(record.mean), torch.tensor(record.scale))
            for record in records
        ],
        "experiment": experiment,
        "view": get(category="data/views", name="Ordered")(config=experiment.data.view),
        "task": get(category="tasks", name="Forecasting")(config=experiment.task),
        "device": torch.device("cpu"),
    }
    return model, protocol, arguments


def test_zero_online_steps_reproduce_frozen_scores(workspace_tmp_path: Path) -> None:
    _, protocol, arguments = _setup(workspace_tmp_path=workspace_tmp_path, steps=0)
    metrics, _ = protocol.evaluate(**arguments)
    assert metrics["online_updates"] == 0.0
    assert metrics["pooled_mse"] == metrics["frozen_pooled_mse"]


def test_prequential_updates_are_causal_and_keep_source_model(
    workspace_tmp_path: Path,
) -> None:
    model, protocol, arguments = _setup(workspace_tmp_path=workspace_tmp_path, steps=1)
    source = {key: value.clone() for key, value in model.state_dict().items()}
    # Count from the window positions alone: before a batch is predicted, exactly
    # the client's earlier windows whose targets end by its origin are trained on.
    stream: dict[str, Any] = {"ends": [], "trained": 0, "checked": 0}
    start_client = protocol.start_client
    release = protocol.regime.release
    predict = protocol.predict

    def recording_start(**kwargs: Any) -> None:
        if protocol.streaming:
            stream.update(ends=[], trained=0)
        start_client(**kwargs)

    def recording_release(
        *, model: nn.Module, observed: int, loss_function: Any
    ) -> None:
        before = len(protocol.regime.pending)
        release(model=model, observed=observed, loss_function=loss_function)
        stream["trained"] += before - len(protocol.regime.pending)

    def recording_predict(**kwargs: Any) -> Tensor:
        windows = kwargs["windows"]
        prediction = predict(**kwargs)
        if not protocol.streaming:
            return prediction
        # Inputs of window (start, target) end at start + input_length (8); targets end
        # at target + output_length (3).
        origin = min(start for start, _ in windows) + 8
        assert stream["trained"] == sum(end <= origin for end in stream["ends"])
        stream["ends"].extend(target + 3 for _, target in windows)
        stream["checked"] += 1
        return prediction

    protocol.start_client = recording_start
    protocol.regime.release = recording_release
    protocol.predict = recording_predict
    metrics, _ = protocol.evaluate(**arguments)
    assert metrics["online_updates"] > 0
    assert metrics["pooled_mse"] != metrics["frozen_pooled_mse"]
    assert stream["checked"] > 2 and stream["trained"] > 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, source[key]), key


def test_reservoir_keeps_a_uniform_bounded_sample() -> None:
    from kairots.replay.Reservoir import Reservoir

    early = 0
    for seed in range(200):
        buffer = Reservoir(config=Reservoir.Config(capacity=10, seed=seed))
        buffer.admit(items=list(range(100)))
        assert len(buffer.items) == 10 and len(set(buffer.items)) == 10
        early += sum(item < 50 for item in buffer.items)
    # Each item is kept with chance 10/100, so half the buffer comes from each half.
    assert abs(early / 2000 - 0.5) < 0.05
    buffer = Reservoir(config=Reservoir.Config(capacity=10))
    buffer.admit(items=list(range(100)))
    assert len(buffer.sample(count=3)) == 3 and buffer.sample(count=99) == buffer.items
    buffer.reset()
    assert buffer.items == [] and buffer.seen == 0


def test_online_replays_only_released_windows() -> None:
    from kairots.training.Online import Online

    model = _Mean()
    regime = Online(config=Online.Config(steps=1, capacity=2, replay_size=2))
    assert regime.replay is not None
    regime.start(optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    trained: list[int] = []

    def loss_function(*, prediction: Tensor, target: Tensor) -> Tensor:
        trained.append(len(target))
        return (prediction - target).pow(2).mean()

    ends = [10, 11, 12, 13]
    regime.observe(batch_x=torch.ones(4, 4, 1), batch_y=torch.zeros(4, 1, 1), ends=ends)
    regime.release(model=model, observed=10, loss_function=loss_function)
    assert trained == [1] and [item[2] for item in regime.replay.items] == [10]
    regime.release(model=model, observed=12, loss_function=loss_function)
    # Two due windows plus the one buffered window.
    assert trained == [1, 3]
    assert all(item[2] <= 12 for item in regime.replay.items)
    regime.start(optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    assert regime.replay.items == []


def test_online_replay_is_optional(workspace_tmp_path: Path) -> None:
    scores = {}
    for index, replay_size in enumerate((0, 0, 32)):
        regime = "Online"
        _, protocol, arguments = _setup(
            workspace_tmp_path=workspace_tmp_path / str(index), steps=1
        )
        protocol.regime = get(category="training", name=regime)(
            config=get(category="training", name=regime).Config(
                steps=1, replay_size=replay_size
            )
        )
        metrics, _ = protocol.evaluate(**arguments)
        scores[len(scores)] = metrics
    online, plain, replayed = scores.values()
    assert online["pooled_mse"] == plain["pooled_mse"]
    assert replayed["pooled_mse"] != online["pooled_mse"]
    # Retention is scored on validation windows, against the frozen model.
    assert online["forgetting_pooled_mse"] == pytest.approx(
        online["retained_pooled_mse"]
        - replayed["retained_pooled_mse"]
        + replayed["forgetting_pooled_mse"]
    )
