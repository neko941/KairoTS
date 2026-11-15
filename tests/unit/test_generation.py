from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from kairots.metrics.ACFError import ACFError
from kairots.metrics.CorrelationError import CorrelationError
from kairots.metrics.DCR import DCR
from kairots.metrics.DiscriminativeScore import DiscriminativeScore
from kairots.metrics.ESError import ESError
from kairots.metrics.KurtosisError import KurtosisError
from kairots.metrics.LeverageError import LeverageError
from kairots.metrics.MMD import MMD
from kairots.metrics.TSTR import TSTR
from kairots.metrics.VaRError import VaRError
from kairots.metrics.VolatilityClusteringError import VolatilityClusteringError
from kairots.metrics.Wasserstein import Wasserstein
from kairots.models.Denoiser import Denoiser
from kairots.tasks.Generation import Generation


def test_prepare_noises_the_window_and_targets_the_noise() -> None:
    task = Generation(config=Generation.Config(diffusion_steps=50))
    batch_y = torch.randn(4, 6, 3)

    def prepare(*, seed: int) -> tuple[Tensor, Tensor]:
        return task.prepare(
            batch_x=batch_y[:, :0],
            batch_y=batch_y,
            output_length=6,
            generator=torch.Generator().manual_seed(seed),
        )

    model_input, noise = prepare(seed=1)
    assert model_input.shape == (4, 6, 4) and noise.shape == batch_y.shape
    steps = model_input[:, :, -1]
    assert (steps == steps[:, :1]).all()
    alpha_bar = task.alpha_bars[steps[:, 0].long()].float()[:, None, None]
    expected = alpha_bar.sqrt() * batch_y + (1 - alpha_bar).sqrt() * noise
    assert torch.allclose(model_input[..., :-1], expected, atol=1e-6)
    again, same_noise = prepare(seed=1)
    assert torch.equal(again, model_input) and torch.equal(same_noise, noise)
    assert not torch.equal(prepare(seed=2)[1], noise)
    with pytest.raises(ValueError, match="generator"):
        task.prepare(batch_x=batch_y[:, :0], batch_y=batch_y, output_length=6)


def test_reverse_process_recovers_a_point_distribution() -> None:
    """With the exact noise predictor of a one-point data distribution, the reverse
    process lands on that point."""
    task = Generation(config=Generation.Config(diffusion_steps=200))
    point = torch.tensor([[1.5, -2.0]]).expand(5, 2)

    class Oracle(nn.Module):
        def forward(self, *, batch_x: Tensor) -> Tensor:
            alpha_bar = task.alpha_bars[batch_x[:, 0, -1].long()].float()
            alpha_bar = alpha_bar[:, None, None]
            return (batch_x[..., :-1] - alpha_bar.sqrt() * point) / (
                1 - alpha_bar
            ).sqrt()

    samples = task.sample(
        model=Oracle(),
        count=3,
        shape=(5, 2),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    assert torch.allclose(samples, point.expand(3, 5, 2), atol=1e-3)


def test_denoiser_maps_a_noised_window_to_its_shape() -> None:
    model = Denoiser(config=Denoiser.Config(input_length=6, input_channels=3, width=8))
    assert model(batch_x=torch.randn(2, 6, 4)).shape == (2, 6, 3)
    with pytest.raises(ValueError, match="even"):
        Denoiser.Config(width=7)


def test_sample_metrics_score_known_distances_and_merge() -> None:
    real = torch.randn(6, 8, 2, generator=torch.Generator().manual_seed(0))
    metric_types: tuple[Any, ...] = (MMD, Wasserstein, ACFError, CorrelationError)
    for metric_type in metric_types:
        metric = metric_type()
        metric.update(prediction=real, target=real)
        assert metric.compute() == pytest.approx(0.0, abs=1e-9)
    # Every value shifted by 1: the marginal distance is 1, the shape unchanged.
    shifted = Wasserstein()
    shifted.update(prediction=real + 1, target=real)
    assert shifted.compute() == pytest.approx(1.0)
    unchanged = ACFError()
    unchanged.update(prediction=real * 3 + 1, target=real)
    # float32 rounding of the affine copy.
    assert unchanged.compute() == pytest.approx(0.0, abs=1e-6)
    # Two equal channels against two opposite ones: off-diagonals 1 and -1.
    base = real[..., :1]
    correlated = CorrelationError()
    correlated.update(
        prediction=torch.cat([base, -base], dim=-1),
        target=torch.cat([base, base], dim=-1),
    )
    assert correlated.compute() == pytest.approx(2 * math.sqrt(2))
    # Merging two halves equals scoring them together.
    generated = torch.randn(6, 8, 2, generator=torch.Generator().manual_seed(1))
    for metric_type in metric_types:
        whole, first, second = metric_type(), metric_type(), metric_type()
        whole.update(prediction=generated, target=real)
        first.update(prediction=generated[:3], target=real[:3])
        second.update(prediction=generated[3:], target=real[3:])
        first.merge(other=second)
        assert first.compute() == pytest.approx(whole.compute())


def test_dcr_flags_copied_train_windows() -> None:
    train = torch.randn(10, 4, 2, generator=torch.Generator().manual_seed(0))
    held_out = torch.randn(3, 4, 2, generator=torch.Generator().manual_seed(1))
    copied = DCR()
    copied.update(prediction=train[:3], target=held_out, train=train)
    assert copied.compute() == 0.0
    fresh = DCR()
    fresh.update(prediction=held_out, target=held_out, train=train)
    assert fresh.compute() == pytest.approx(1.0)


def _ar1(*, count: int, seed: int) -> Tensor:
    """`count` 16-step windows of two AR(1) channels (coefficients 0.9 and -0.5)."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(count, 16, 2, generator=generator)
    windows = torch.zeros(count, 16, 2)
    for step in range(1, 16):
        windows[:, step] = (
            torch.tensor([0.9, -0.5]) * windows[:, step - 1] + noise[:, step]
        )
    return windows


def test_tstr_matches_real_training_only_for_the_same_process() -> None:
    train, held_out = _ar1(count=200, seed=0), _ar1(count=50, seed=1)
    same = TSTR()
    same.update(prediction=_ar1(count=200, seed=2), target=held_out, train=train)
    assert same.compute() == pytest.approx(1.0, abs=0.1)
    shuffled = TSTR()
    # White noise of the same scale carries no autoregression to learn.
    shuffled.update(
        prediction=torch.randn(200, 16, 2, generator=torch.Generator().manual_seed(3))
        * train.std(),
        target=held_out,
        train=train,
    )
    assert shuffled.compute() > 1.5


def test_discriminative_score_separates_shifted_samples() -> None:
    real = _ar1(count=150, seed=0)
    same = DiscriminativeScore()
    same.update(prediction=_ar1(count=150, seed=1), target=real)
    assert same.compute() < 0.1
    shifted = DiscriminativeScore()
    shifted.update(prediction=real[:75] + 20, target=real[75:])
    assert shifted.compute() == 0.5
    # Exact copies are flagged too: every window's neighbour is its twin.
    copied = DiscriminativeScore()
    copied.update(prediction=real, target=real)
    assert copied.compute() == 0.5


def _garch(*, count: int, seed: int) -> Tensor:
    """`count` 24-step windows of a GARCH(1, 1)-like series whose volatility rises
    after a fall: heavy tails, volatility clustering and a leverage effect."""
    generator = torch.Generator().manual_seed(seed)
    shocks = torch.randn(count, 24, 1, generator=generator)
    values = torch.zeros(count, 24, 1)
    variance = torch.ones(count, 1)
    for step in range(24):
        values[:, step] = variance.sqrt() * shocks[:, step]
        fall = (values[:, step] < 0).double()
        variance = 0.2 + 0.5 * variance + (0.1 + 0.3 * fall) * values[:, step] ** 2
    return values


def test_stylized_facts_tell_a_garch_series_from_gaussian_noise() -> None:
    real, again = _garch(count=2000, seed=0), _garch(count=2000, seed=1)
    noise = torch.randn(2000, 24, 1, generator=torch.Generator().manual_seed(2))
    noise = noise * real.std()
    metric_types: tuple[Any, ...] = (
        KurtosisError,
        VolatilityClusteringError,
        LeverageError,
        VaRError,
        ESError,
    )
    for metric_type in metric_types:
        same, gaussian = metric_type(), metric_type()
        same.update(prediction=again, target=real)
        gaussian.update(prediction=noise, target=real)
        # At 5% a Gaussian of the same variance has a VaR and ES close to the real
        # ones; the known-value test covers those two.
        if metric_type not in (VaRError, ESError):
            assert same.compute() < gaussian.compute(), metric_type.__name__
        # Merging two halves equals scoring them together.
        first, second = metric_type(), metric_type()
        first.update(prediction=noise[:1000], target=real[:1000])
        second.update(prediction=noise[1000:], target=real[1000:])
        first.merge(other=second)
        assert first.compute() == pytest.approx(gaussian.compute())


def test_tail_risk_and_kurtosis_known_values() -> None:
    # Values 1..100 per channel: the 5% quantile is 5.95, the mean below it 3.
    values = torch.arange(1.0, 101.0)[None, :, None].expand(1, 100, 2)
    var, shortfall = VaRError(), ESError()
    for metric in (var, shortfall):
        metric.update(prediction=values + 1, target=values)
    assert var.compute() == pytest.approx(1.0)
    assert shortfall.compute() == pytest.approx(1.0)
    # A two-point distribution has excess kurtosis -2; a Gaussian about 0.
    kurtosis = KurtosisError()
    kurtosis.update(
        prediction=torch.randn(
            1, 100000, 1, generator=torch.Generator().manual_seed(0)
        ),
        target=torch.tensor([-1.0, 1.0]).repeat(50)[None, :, None].expand(1, 100, 1),
    )
    assert kurtosis.compute() == pytest.approx(2.0, abs=0.05)
