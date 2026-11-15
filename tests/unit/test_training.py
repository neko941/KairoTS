"""Continual is shared schedule infrastructure, never a selectable alias."""

import pytest

from kairots.registry import get, info, names
from kairots.training.base import Continual
from kairots.training.ClassIncremental import ClassIncremental


def test_continual_is_a_non_selectable_base() -> None:
    assert "Continual" not in names(category="training")
    with pytest.raises(KeyError, match="Continual"):
        get(category="training", name="Continual")
    assert issubclass(ClassIncremental, Continual)
    assert info(category="training", name="ClassIncremental")["CONTINUAL"] is True
    with pytest.raises(NotImplementedError):
        Continual(config=Continual.Config()).windows(
            windows=[], period=0, num_classes=None
        )


@pytest.mark.parametrize(
    "periods,capacity,replay_size", [(1, 256, 0), (2, 0, 0), (2, 256, -1)]
)
def test_continual_subclasses_keep_shared_config_validation(
    *, periods: int, capacity: int, replay_size: int
) -> None:
    with pytest.raises(ValueError, match="Continual needs"):
        ClassIncremental.Config(
            periods=periods, capacity=capacity, replay_size=replay_size
        )


def test_class_incremental_releases_are_disjoint_and_cover_each_period() -> None:
    schedule = ClassIncremental(config=ClassIncremental.Config(periods=2))
    windows = [(index, index % 4) for index in range(24)]
    releases = [
        schedule.released(
            windows=windows, current_iteration=index, iterations=3, num_classes=4
        )
        for index in range(6)
    ]
    assert sorted(item for release in releases for item in release) == windows
    for index, release in enumerate(releases):
        assert all(label // 2 == index // 3 for _, label in release)
    with pytest.raises(ValueError, match="labeled classification"):
        schedule.windows(windows=windows, period=0, num_classes=None)
