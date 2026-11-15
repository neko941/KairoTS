from __future__ import annotations

from typing import Any

import pytest
import torch

from kairots.strategies.FLwF2 import FLwF2


def test_flwf2_summed_loss_and_gradient_match_the_three_cross_entropies() -> None:
    student = torch.tensor(
        [[0.2, -0.4], [1.0, -0.2]], dtype=torch.float64, requires_grad=True
    )
    server = torch.tensor(
        [[0.8, 0.0], [0.2, 0.9]], dtype=torch.float64, requires_grad=True
    )
    previous = torch.tensor(
        [[-0.6, 0.5], [0.4, 0.6]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([0, 1])
    config = FLwF2.Config(alpha=0.2, beta=0.3, temperature=2)
    for teacher in (None, previous):
        loss = FLwF2.objective(
            prediction=student,
            target=target,
            server=server,
            previous=teacher,
            config=config,
        )
        probabilities = student.detach().softmax(dim=-1)
        softened = (student.detach() / 2).softmax(dim=-1)
        expected = -0.2 * probabilities[torch.arange(2), target].log().sum()
        gradient = 0.2 * (
            probabilities - torch.nn.functional.one_hot(target, num_classes=2)
        )
        for weight, scores in ((0.5, server), (0.3, teacher)):
            if scores is not None:
                teacher_probabilities = (scores.detach() / 2).softmax(dim=-1)
                expected -= weight * (teacher_probabilities * softened.log()).sum()
                gradient += weight * (softened - teacher_probabilities) / 2
        assert loss.item() == pytest.approx(expected.item())
        torch.testing.assert_close(
            torch.autograd.grad(loss, inputs=student)[0], gradient
        )
    assert server.grad is None and previous.grad is None


def test_flwf2_rejects_invalid_weights_and_participation() -> None:
    cases: list[dict[str, Any]] = [
        {"alpha": 0.8, "beta": 0.3},
        {"temperature": 0},
        {"join_ratio": 0.5},
    ]
    for options in cases:
        with pytest.raises(ValueError):
            FLwF2.Config(**options)
