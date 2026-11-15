from types import SimpleNamespace

import pytest
import torch

from kairots.config.schema import Experiment
from kairots.metrics.M4OWA import M4OWA, M4Context, overall_owa


def test_m4_owa_matches_formula_and_merges() -> None:
    prediction = torch.tensor([[[2.0], [0.0]], [[4.0], [2.0]]])
    target = torch.tensor([[[1.0], [1.0]], [[3.0], [4.0]]])
    seasonal_scale = torch.tensor([[2.0], [3.0]])
    context = M4Context(baseline_smape=100.0, baseline_mase=1.0)

    whole = M4OWA()
    whole.update(
        prediction=prediction,
        target=target,
        seasonal_scale=seasonal_scale,
        m4_context=context,
    )
    first, second = M4OWA(), M4OWA()
    for metric, index in ((first, slice(0, 1)), (second, slice(1, 2))):
        metric.update(
            prediction=prediction[index],
            target=target[index],
            seasonal_scale=seasonal_scale[index],
            m4_context=context,
        )
    first.merge(other=second)

    denominator = prediction.abs() + target.abs()
    smape = (200 * (prediction - target).abs() / denominator).mean().item()
    mase = ((prediction - target).abs() / seasonal_scale[:, None, :]).mean().item()
    expected = 0.5 * (smape / context.baseline_smape + mase / context.baseline_mase)
    assert whole.compute() == pytest.approx(expected)
    assert first.compute() == pytest.approx(expected)


def test_m4_owa_resolves_only_matching_official_fold() -> None:
    experiment = SimpleNamespace(
        data=SimpleNamespace(
            source=SimpleNamespace(name="M4Hourly", seasonality=24),
            split=SimpleNamespace(output_length=48),
        )
    )
    context = M4OWA.prepare_context(experiment=experiment)
    assert context.baseline_smape == pytest.approx(18.382878117866)
    assert context.baseline_mase == pytest.approx(2.395040006949)

    experiment.data.source.seasonality = 1
    with pytest.raises(ValueError, match="seasonality"):
        M4OWA.prepare_context(experiment=experiment)
    experiment.data.source.name = "ETTh1"
    with pytest.raises(ValueError, match="official M4"):
        M4OWA.prepare_context(experiment=experiment)


def test_m4_owa_is_test_only() -> None:
    for phase in ("train", "validation"):
        with pytest.raises(ValueError, match=f"cannot be selected for the {phase}"):
            Experiment.from_dict(values={"evaluation": {"metrics": {phase: ["M4OWA"]}}})
    Experiment.from_dict(values={"evaluation": {"metrics": {"test": ["M4OWA"]}}})


def test_overall_m4_owa_weights_folds_by_series_count() -> None:
    smape = {
        "M4Yearly": 16.342173488085,
        "M4Quarterly": 11.011569928825,
        "M4Monthly": 14.426828759880,
        "M4Weekly": 9.161286913982,
        "M4Daily": 3.045251780079,
        "M4Hourly": 18.382878117866,
    }
    mase = {
        "M4Yearly": 3.974360265174,
        "M4Quarterly": 1.371382270977,
        "M4Monthly": 1.063359447971,
        "M4Weekly": 2.777294993197,
        "M4Daily": 3.278424295960,
        "M4Hourly": 2.395040006949,
    }
    assert overall_owa(fold_smape=smape, fold_mase=mase) == pytest.approx(1.0)
    changed = smape.copy()
    changed["M4Yearly"] += 1.0
    counts = {
        "M4Yearly": 23000,
        "M4Quarterly": 24000,
        "M4Monthly": 48000,
        "M4Weekly": 359,
        "M4Daily": 4227,
        "M4Hourly": 414,
    }
    expected = 0.5 * (
        sum(counts[name] * changed[name] for name in counts)
        / sum(counts[name] * smape[name] for name in counts)
        + 1.0
    )
    assert overall_owa(fold_smape=changed, fold_mase=mase) == pytest.approx(expected)
    with pytest.raises(ValueError, match="all six frequency folds"):
        overall_owa(fold_smape={"M4Yearly": 0.0}, fold_mase=mase)
