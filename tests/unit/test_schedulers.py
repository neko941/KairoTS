from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from kairots.schedulers.Constant import Constant
from kairots.schedulers.Cosine import Cosine
from kairots.schedulers.OneCycle import OneCycle


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=1.0)


def test_cosine_follows_closed_form_and_resumes_from_state() -> None:
    optimizer = _optimizer()
    scheduler = Cosine(config=Cosine.Config()).build(optimizer=optimizer, iterations=4)
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    # A client's next round rebuilds both from its saved state.
    resumed_optimizer = _optimizer()
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed = Cosine(config=Cosine.Config()).build(
        optimizer=resumed_optimizer, iterations=4
    )
    resumed.load_state_dict(scheduler.state_dict())
    resumed_optimizer.step()
    resumed.step()
    expected = 0.5 * (1 + math.cos(math.pi * 3 / 4))
    assert math.isclose(resumed_optimizer.param_groups[0]["lr"], expected)


def test_constant_keeps_the_learning_rate() -> None:
    optimizer = _optimizer()
    scheduler = Constant(config=Constant.Config()).build(
        optimizer=optimizer, iterations=4
    )
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0


def test_one_cycle_advances_per_optimizer_step() -> None:
    optimizer = _optimizer()
    scheduler = OneCycle(config=OneCycle.Config(maximum_rate=0.1)).build(
        optimizer=optimizer, iterations=100
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert max(rates) == pytest.approx(0.1)
    assert rates[-1] < rates[0]


def test_one_cycle_rejects_empty_training_budget() -> None:
    optimizer = _optimizer()
    with pytest.raises(ValueError, match="at least one optimizer step"):
        OneCycle(config=OneCycle.Config()).build(optimizer=optimizer, iterations=0)


def test_one_cycle_rejects_nonpositive_maximum_rate() -> None:
    with pytest.raises(ValueError, match="maximum_rate must be positive"):
        OneCycle.Config(maximum_rate=0)


def test_one_cycle_resume_continues_at_the_saved_rate() -> None:
    optimizer = _optimizer()
    scheduler = OneCycle(config=OneCycle.Config()).build(
        optimizer=optimizer, iterations=8
    )
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    expected_rate = optimizer.param_groups[0]["lr"]

    resumed_optimizer = _optimizer()
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed = OneCycle(config=OneCycle.Config()).build(
        optimizer=resumed_optimizer, iterations=8
    )
    resumed.load_state_dict(scheduler.state_dict())
    for group, rate in zip(
        resumed_optimizer.param_groups, resumed.get_last_lr(), strict=True
    ):
        group["lr"] = rate
    assert resumed_optimizer.param_groups[0]["lr"] == expected_rate

    optimizer.step()
    scheduler.step()
    resumed_optimizer.step()
    resumed.step()
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )


def test_cosine_decay_iterations_then_holds_minimum() -> None:
    optimizer = _optimizer()
    scheduler = Cosine(config=Cosine.Config(decay_iterations=2)).build(
        optimizer=optimizer, iterations=10
    )
    rates = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert math.isclose(rates[0], 0.5) and rates[1:] == [0.0, 0.0, 0.0]
    # The worker rebuilds the schedule every update on the annealed optimizer.
    state = scheduler.state_dict()
    rebuilt = Cosine(config=Cosine.Config(decay_iterations=2)).build(
        optimizer=optimizer, iterations=10
    )
    rebuilt.load_state_dict(state)
    optimizer.step()
    rebuilt.step()
    assert optimizer.param_groups[0]["lr"] == 0.0
