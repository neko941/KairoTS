from __future__ import annotations

import pytest
import torch

from kairots.config.schema import Experiment
from kairots.metrics.MAE import MAE


def test_mae_batch_merge_equals_pooled_value() -> None:
    prediction = torch.tensor([1.0, 4.0, 7.0])
    target = torch.tensor([0.0, 2.0, 9.0])
    first, second, pooled = MAE(), MAE(), MAE()
    first.update(prediction=prediction[:1], target=target[:1])
    second.update(prediction=prediction[1:], target=target[1:])
    pooled.update(prediction=prediction, target=target)
    first.merge(other=second)
    assert first.compute() == pytest.approx(pooled.compute())


def test_metrics_and_losses_declare_where_they_apply() -> None:
    from kairots.registry import info, names, require

    for name in names(category="metrics"):
        constants = info(category="metrics", name=name)
        assert constants.get("TASKS") and constants.get("NEEDS"), name
        assert set(constants["ROLES"]) <= {"loss", "metric"}, name
        if "metric" in constants["ROLES"]:
            assert constants.get("DIRECTION") in {"min", "max"}, name
    assert names(category="metrics", task="Forecasting") == (
        "M4OWA",
        "M5WRMSSE",
        "MAE",
        "MASE",
        "MSE",
        "R2",
        "SMAPE",
    )
    assert names(category="metrics", task="Classification") == (
        "AUROC",
        "Accuracy",
        "CrossEntropy",
        "MacroF1",
    )
    assert names(category="metrics", task="ProbabilisticForecasting") == (
        "CRPS",
        "CoverageError",
        "IntervalWidth",
        "Pinball",
    )
    assert names(category="metrics", task="Generation") == (
        "ACFError",
        "CorrelationError",
        "DCR",
        "DiscriminativeScore",
        "ESError",
        "HillError",
        "KurtosisError",
        "LeverageError",
        "MMD",
        "MSE",
        "TSTR",
        "VaRError",
        "VolatilityClusteringError",
        "Wasserstein",
    )
    with pytest.raises(ValueError, match="does not support task 'Classification'"):
        require(category="metrics", name="MAE", task="Classification")


def test_r2_merges_exactly_and_is_test_only() -> None:
    from kairots.metrics.R2 import R2

    generator = torch.Generator().manual_seed(0)
    target = torch.randn(6, 4, 3, generator=generator)
    prediction = target + 0.3 * torch.randn(6, 4, 3, generator=generator)
    first, second, pooled = R2(), R2(), R2()
    first.update(prediction=prediction[:2], target=target[:2])
    second.update(prediction=prediction[2:], target=target[2:])
    pooled.update(prediction=prediction, target=target)
    first.merge(other=second)
    flat_target = target.flatten(end_dim=-2).double()
    flat_prediction = prediction.flatten(end_dim=-2).double()
    reference = 1 - ((flat_prediction - flat_target) ** 2).sum(dim=0) / (
        (flat_target - flat_target.mean(dim=0)) ** 2
    ).sum(dim=0)
    assert first.compute() == pytest.approx(pooled.compute())
    assert pooled.compute() == pytest.approx(reference.mean().item())
    # Unscored (NaN) targets drop out per channel.
    masked = R2()
    masked.update(
        prediction=torch.tensor([[1.0], [5.0], [3.0]]),
        target=torch.tensor([[1.0], [float("nan")], [3.0]]),
    )
    assert masked.compute() == 1.0
    for phase in ("train", "validation"):
        with pytest.raises(ValueError, match=f"cannot be selected for the {phase}"):
            Experiment.from_dict(
                values={"evaluation": {"metrics": {phase: ["MAE", "R2"]}}}
            )
    Experiment.from_dict(values={"evaluation": {"metrics": {"test": ["R2"]}}})


def test_smape_matches_tslib_and_merges() -> None:
    import numpy as np

    from kairots.metrics.SMAPE import SMAPE

    prediction = torch.tensor([[1.0, 0.0, 3.0], [2.0, 5.0, -1.0]])
    target = torch.tensor([[2.0, 0.0, 1.0], [2.0, 4.0, 1.0]])
    forecast, outsample = prediction.numpy(), target.numpy()
    denominator = np.abs(outsample) + np.abs(forecast)
    denominator[denominator == 0.0] = 1.0
    expected = np.mean(200 * np.abs(forecast - outsample) / denominator)
    whole, first, second = SMAPE(), SMAPE(), SMAPE()
    whole.update(prediction=prediction, target=target)
    first.update(prediction=prediction[:1], target=target[:1])
    second.update(prediction=prediction[1:], target=target[1:])
    first.merge(other=second)
    assert abs(whole.compute() - float(expected)) < 1e-5
    assert abs(first.compute() - whole.compute()) < 1e-5


def test_crps_from_quantiles_matches_point_mae_and_gaussian_closed_form() -> None:
    from kairots.metrics.CRPS import CRPS
    from kairots.metrics.Pinball import Pinball

    target = torch.tensor([[0.5, -1.0], [2.0, float("nan")]])
    point = torch.tensor([[0.0, 0.0], [1.0, 9.0]])
    one_level = CRPS()
    one_level.update(prediction=point.unsqueeze(-1), target=target)
    mae = MAE()
    mae.update(prediction=point, target=target)
    assert one_level.compute() == pytest.approx(mae.compute())
    loss = Pinball(config=Pinball.Config())
    assert float(loss(prediction=point.unsqueeze(-1), target=target)) == (
        pytest.approx(mae.compute() / 2)
    )

    # N(0, 1) quantiles at 999 levels against y: closed-form Gaussian CRPS.
    num_quantiles = 999
    levels = torch.arange(1, num_quantiles + 1, dtype=torch.float64) / 1000
    normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    observed = torch.tensor([0.3, -1.2], dtype=torch.float64)
    quantiles = normal.icdf(levels).expand(2, num_quantiles)
    first, second = CRPS(), CRPS()
    first.update(prediction=quantiles[:1], target=observed[:1])
    second.update(prediction=quantiles[1:], target=observed[1:])
    first.merge(other=second)
    expected = (
        observed * (2 * normal.cdf(observed) - 1)
        + 2 * normal.log_prob(observed).exp()
        - 1 / torch.pi**0.5
    ).mean()
    assert first.compute() == pytest.approx(float(expected), abs=1e-3)
    with pytest.raises(ValueError, match="plus a quantile axis"):
        loss(prediction=point, target=target)


def test_mase_scales_by_seasonal_difference_and_merges() -> None:
    from kairots.metrics.MASE import MASE, seasonal_scales

    series = torch.tensor([[0.0], [1.0], [2.0], [3.0], [5.0]])
    # Targets at row 4 (history 0..3) and row 5 (history 0..4).
    scale = seasonal_scales(series=series, target_rows=[4, 5], seasonality=1)
    assert torch.allclose(scale, torch.tensor([[1.0], [5.0 / 4]]))
    target = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    metric = MASE()
    metric.update(
        prediction=target + 2, target=target, seasonal_scale=scale[:1].repeat(2, 1)
    )
    assert metric.compute() == 2.0
    # A per-channel affine scaler changes nothing.
    scaled = MASE()
    scaled.update(
        prediction=(target + 2) * 5 - 3,
        target=target * 5 - 3,
        seasonal_scale=seasonal_scales(
            series=series * 5 - 3, target_rows=[4, 4], seasonality=1
        ),
    )
    assert abs(scaled.compute() - 2.0) < 1e-6
    first, second = MASE(), MASE()
    first.update(prediction=target[:1] + 2, target=target[:1], seasonal_scale=scale[:1])
    second.update(
        prediction=target[1:] + 1, target=target[1:], seasonal_scale=scale[:1]
    )
    first.merge(other=second)
    assert first.compute() == 1.5
    constant = MASE()
    constant.update(
        prediction=target + 3,
        target=target,
        seasonal_scale=seasonal_scales(
            series=torch.ones(4, 1), target_rows=[4, 4], seasonality=2
        ),
    )
    assert constant.compute() == 3.0
    with pytest.raises(ValueError, match="longer than seasonality"):
        seasonal_scales(series=series, target_rows=[1], seasonality=1)


def test_interval_coverage_error_and_width_merge() -> None:
    from kairots.metrics.CoverageError import CoverageError
    from kairots.metrics.IntervalWidth import IntervalWidth

    # Three levels (0.25, 0.5, 0.75): the outer interval claims 50% coverage.
    prediction = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 1.0, 4.0]]])
    target = torch.tensor([[1.0, 5.0]])
    first, second = CoverageError(), CoverageError()
    first.update(prediction=prediction[:, :1], target=target[:, :1])
    second.update(prediction=prediction[:, 1:], target=target[:, 1:])
    first.merge(other=second)
    assert first.compute() == 0.0
    missing = CoverageError()
    missing.update(prediction=prediction, target=torch.tensor([[1.0, float("nan")]]))
    assert missing.compute() == 0.5
    width = IntervalWidth()
    width.update(prediction=prediction, target=target)
    assert width.compute() == 3.0
    with pytest.raises(ValueError, match="num_quantiles >= 2"):
        CoverageError().update(prediction=prediction[..., :1], target=target)


def test_auroc_merges_and_ranks_ties() -> None:
    import torch

    from kairots.metrics.AUROC import AUROC

    scores = torch.tensor([[0.0, s] for s in (0.1, 0.4, 0.4, 0.8, 0.3)])
    labels = torch.tensor([0, 0, 1, 1, 1])
    whole, first, second = AUROC(), AUROC(), AUROC()
    whole.update(prediction=scores, target=labels)
    first.update(prediction=scores[:2], target=labels[:2])
    second.update(prediction=scores[2:], target=labels[2:])
    first.merge(other=second)
    # Six positive-negative pairs: 4 won, 1 tied (half), 1 lost.
    assert whole.compute() == first.compute() == 0.75
