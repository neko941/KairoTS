from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kairots.config.schema import Experiment
from kairots.metrics.MAE import MAE
from kairots.metrics.MSE import MSE
from kairots.tasks.Imputation import Imputation


def _prepare(
    *, batch_y: torch.Tensor, seed: int, ratio: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    task = Imputation(config=Imputation.Config(mask_ratio=ratio))
    return task.prepare(
        batch_x=batch_y[:, :0],
        batch_y=batch_y,
        output_length=batch_y.shape[1],
        generator=torch.Generator().manual_seed(seed),
    )


def test_hidden_values_never_reach_the_model_input() -> None:
    batch_y = torch.randn(4, 16, 3, generator=torch.Generator().manual_seed(0))
    model_input, target = _prepare(batch_y=batch_y, seed=1)
    hidden = target.isfinite()
    assert hidden.any() and (~hidden).any()
    assert (model_input[hidden] == 0).all()
    assert torch.equal(model_input[~hidden], batch_y[~hidden])
    # Changing hidden values changes the target, never the input.
    changed = batch_y.clone()
    changed[hidden] += 100.0
    same_input, _ = _prepare(batch_y=changed, seed=1)
    assert torch.equal(same_input, model_input)


def test_scoring_masks_are_reproducible_and_seeded() -> None:
    batch_y = torch.randn(2, 8, 2)
    _, first = _prepare(batch_y=batch_y, seed=5)
    _, again = _prepare(batch_y=batch_y, seed=5)
    _, other = _prepare(batch_y=batch_y, seed=6)
    assert torch.equal(first.isfinite(), again.isfinite())
    assert not torch.equal(first.isfinite(), other.isfinite())


def test_native_missing_values_are_neither_input_nor_scored() -> None:
    batch_y = torch.ones(1, 32, 2)
    batch_y[0, :8, 0] = torch.nan
    model_input, target = _prepare(batch_y=batch_y, seed=0, ratio=0.5)
    assert model_input.isfinite().all()
    assert (model_input[0, :8, 0] == 0).all()
    assert not target[0, :8, 0].isfinite().any()


def test_empty_supervision_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="no observed positions"):
        _prepare(batch_y=torch.full((1, 4, 1), torch.nan), seed=0)


def test_a_draw_that_hides_nothing_hides_one_observed_position() -> None:
    batch_y = torch.full((1, 4, 1), torch.nan)
    batch_y[0, 2, 0] = 7.0
    # A tiny ratio hides nothing, so the one observed value is hidden instead.
    model_input, target = _prepare(batch_y=batch_y, seed=0, ratio=1e-9)
    assert target[0, 2, 0] == 7.0 and target.isfinite().sum() == 1
    assert model_input[0, 2, 0] == 0


def test_loss_and_metric_score_only_finite_targets() -> None:
    prediction = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, torch.nan, 5.0]])
    assert MSE(config=MSE.Config())(prediction=prediction, target=target) == 2.0
    metric = MAE()
    metric.update(prediction=prediction, target=target)
    assert metric.compute() == 1.0


def test_imputation_config_feeds_the_target_window_to_the_model() -> None:
    values = json.loads(
        (
            Path(__file__).parents[2] / "configs/examples/imputation/ETDatasetHour.json"
        ).read_text(encoding="utf-8")
    )
    experiment = Experiment.from_dict(values=values)
    assert experiment.lengths() == (0, 96, 0)
    assert experiment.model.input_length == experiment.model.output_length == 96
    assert experiment.model.head == "Forecasting"
    assert experiment.evaluation.loss == "MSE"
