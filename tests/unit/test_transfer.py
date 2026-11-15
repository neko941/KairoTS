from __future__ import annotations

from pathlib import Path

import torch
from fixtures.synthetic import ett_experiment, write_ett_sources

from kairots.config.schema import Experiment
from kairots.data.stores.Parquet import Parquet
from kairots.learners.Gradient import Gradient
from kairots.metrics.MSE import MSE
from kairots.models.DLinear import DLinear
from kairots.strategies.Transfer import Transfer


def test_frozen_parameters_keep_their_values() -> None:
    torch.manual_seed(0)
    model = DLinear(
        config=DLinear.Config(input_length=8, output_length=3, input_channels=2)
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    frozen = [name for name in before if "linear_seasonal" in name]
    assert frozen
    learner = Gradient(config=Gradient.Config(learning_rate=0.1))
    Transfer.Client(frozen=("*linear_seasonal.*",)).fit(
        learner=learner,
        model=model,
        optimizer=learner.optimizer(model=model),
        batches=[(torch.randn(4, 8, 2), torch.randn(4, 3, 2))],
        epochs=2,
        device=torch.device("cpu"),
        loss_function=MSE(config=MSE.Config()),
        reference_parameters={},
    )
    for name, value in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, before[name])
        else:
            assert not torch.equal(value, before[name])


def test_train_windows_keeps_the_last_windows_before_the_cut(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    values = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs"
    ).to_dict()

    def splits(*, budget: int) -> list[dict[str, list[tuple[int, int]]]]:
        values["data"]["split"]["train_windows"] = budget
        experiment = Experiment.from_dict(values=values)
        _, records = Parquet(config=experiment.data.store).prepare(
            experiment=experiment
        )
        return [Parquet.load(record=record)[1] for record in records]

    full, budgeted = splits(budget=0), splits(budget=5)
    for every, few in zip(full, budgeted, strict=True):
        assert [tuple(window) for window in few["train"]] == [
            tuple(window) for window in every["train"][-5:]
        ]
        assert [tuple(window) for window in few["test"]] == [
            tuple(window) for window in every["test"]
        ]
