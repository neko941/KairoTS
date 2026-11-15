from __future__ import annotations

import math

import pytest
import torch

from kairots.metrics.HillError import HillError, hill


def test_hill_uses_the_next_order_statistic_and_is_scale_invariant() -> None:
    values = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    expected = 1.5 * math.log(2)
    assert hill(values=values, tail_count=2).item() == pytest.approx(expected)
    assert hill(values=values * 100, tail_count=2).item() == pytest.approx(expected)
    # Squaring doubles xi; taking its reciprocal here would fail this check.
    assert hill(values=values.square(), tail_count=2).item() == pytest.approx(
        2 * expected
    )
    config = HillError.Config(tail_count=2)
    first, second, whole = (HillError(config=config) for _ in range(3))
    real = values.reshape(2, 2, 1)
    generated = -real.square()
    first.update(prediction=generated[:1], target=real[:1])
    second.update(prediction=generated[1:], target=real[1:])
    whole.update(prediction=generated, target=real)
    first.merge(other=second)
    assert first.compute() == pytest.approx(expected)
    assert first.compute() == pytest.approx(whole.compute())
    for invalid in (values * 0, -values, values * float("nan")):
        with pytest.raises(ValueError):
            hill(values=invalid, tail_count=2)
    for count in (0, 4):
        with pytest.raises(ValueError):
            hill(values=values, tail_count=count)
