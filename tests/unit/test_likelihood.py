from __future__ import annotations

import math

import pytest
import torch

from kairots.metrics.GaussianNLL import GaussianNLL
from kairots.models.DLinear import DLinear


def test_gaussian_likelihood_matches_density_and_merges_masked_positions() -> None:
    prediction = torch.tensor([[0.0, 1.0], [2.0, 0.5], [5.0, 1.0]], requires_grad=True)
    target = torch.tensor([1.0, 2.5, float("nan")])
    expected = (math.log(2 * math.pi) + 1 + math.log(0.5)) / 2
    loss = GaussianNLL(config=GaussianNLL.Config())(
        prediction=prediction, target=target
    )
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert prediction.grad is not None and prediction.grad.isfinite().all()
    assert torch.equal(prediction.grad[2], torch.zeros(2))
    first, second, whole = GaussianNLL(), GaussianNLL(), GaussianNLL()
    first.update(prediction=prediction[:1], target=target[:1])
    second.update(prediction=prediction[1:], target=target[1:])
    whole.update(prediction=prediction, target=target)
    first.merge(other=second)
    assert first.compute() == pytest.approx(whole.compute())
    model = DLinear(
        config=DLinear.Config(head="Gaussian", input_length=8, output_length=3)
    )
    forecast = model(batch_x=torch.randn(2, 8, 7))
    assert forecast.shape == (2, 3, 7, 2) and (forecast[..., 1] > 0).all()
    with pytest.raises(ValueError):
        GaussianNLL(config=GaussianNLL.Config())(
            prediction=prediction.detach() * 0, target=target
        )
