from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from kairots.config.schema import Experiment
from kairots.evaluation.Frozen import Frozen
from kairots.evaluation.populations import select
from kairots.registry import get


def test_population_selection_reuses_seen_and_unseen_clients() -> None:
    selected = select(
        client_ids=[0, 1, 2],
        unseen_client_ids={2},
        phases=["validation", "test"],
        cross_dataset=True,
    )
    assert selected == {
        "cross_": ([0, 1], ["validation", "test"]),
        "unseen_": ([2], ["test"]),
    }


def test_population_selection_rejects_unknown_unseen_client() -> None:
    with pytest.raises(ValueError, match="loaded population"):
        select(
            client_ids=[0],
            unseen_client_ids={1},
            phases=["test"],
            cross_dataset=False,
        )


class _Zero(nn.Module):
    def __init__(self, *, output_length: int) -> None:
        super().__init__()
        self.output_length = output_length

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return batch_x.new_zeros(batch_x.shape[0], self.output_length, batch_x.shape[2])


def _scores(*, inverse: bool, phase: str) -> dict[str, float]:
    experiment = Experiment.from_dict(
        values={
            "data": {
                "split": {
                    "input_length": 4,
                    "output_length": 2,
                    "ratios": [0.6, 0.2, 0.2],
                }
            },
            "evaluation": {"inverse": inverse, "metrics": {"test": ["MAE"]}},
        }
    )
    values = torch.linspace(-1.0, 1.0, 30)[:, None]
    windows = [(start, start + 4) for start in range(24)]
    loss_type = get(category="metrics", name="MSE")
    metrics, _ = Frozen(
        config=experiment.evaluation, loss_function=loss_type(config=loss_type.Config())
    ).evaluate(
        phase=phase,
        model=_Zero(output_length=2),
        clients=[(values, {"train": [], "validation": windows, "test": windows})],
        scales=[(torch.tensor([5.0]), torch.tensor([3.0]))],
        experiment=experiment,
        view=get(category="data/views", name="Offline")(config=experiment.data.view),
        task=get(category="tasks", name="Forecasting")(config=experiment.task),
        device=torch.device("cpu"),
    )
    return metrics


def test_inverse_scores_test_in_raw_units_and_leaves_validation_scaled() -> None:
    scaled, raw = (
        _scores(inverse=False, phase="test"),
        _scores(inverse=True, phase="test"),
    )
    # A zero prediction is the mean in raw units: errors grow by the scale (3).
    assert abs(raw["pooled_mae"] - 3 * scaled["pooled_mae"]) < 1e-5
    assert abs(raw["pooled_mse"] - 9 * scaled["pooled_mse"]) < 1e-4
    assert _scores(inverse=True, phase="validation") == _scores(
        inverse=False, phase="validation"
    )


def test_bottom_up_forecasts_add_up_in_raw_units() -> None:
    from kairots.evaluation.BottomUp import bottom_up

    # Channels: Total, A, B; Total = A + B.
    matrix = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    mean, scale = torch.tensor([10.0, 4.0, 6.0]), torch.tensor([5.0, 2.0, 3.0])
    prediction = torch.randn(4, 12, 3, generator=torch.Generator().manual_seed(0))
    raw = (
        bottom_up(
            prediction=prediction,
            mean=mean,
            scale=scale,
            matrix=matrix,
            bottom=torch.tensor([1, 2]),
        )
        * scale
        + mean
    )
    assert torch.allclose(raw[..., 0], raw[..., 1] + raw[..., 2], atol=1e-5)
    assert torch.allclose(raw[..., 1:], prediction[..., 1:] * scale[1:] + mean[1:])
