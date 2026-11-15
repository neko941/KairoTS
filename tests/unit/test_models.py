from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def test_model_checkpoint_keys_are_preserved() -> None:
    from kairots.registry import get, names

    expected = json.loads(
        (Path(__file__).parents[1] / "fixtures/model_state_keys.json").read_text()
    )
    assert set(expected) == set(names(category="models"))
    for name, keys in expected.items():
        recipe = get(category="models", name=name)
        assert list(recipe(config=recipe.Config()).state_dict()) == keys


def test_dlinear_heads_share_the_body_and_match_reference() -> None:
    from torch.nn import functional

    from kairots.models.DLinear import DLinear

    torch.manual_seed(0)
    batch_x = torch.randn(4, 96, 7)
    forecaster = DLinear(config=DLinear.Config(individual=True))
    assert forecaster(batch_x=batch_x).shape == (4, 24, 7)
    quantile = DLinear(config=DLinear.Config(head="Quantile", num_quantiles=5))
    assert quantile(batch_x=batch_x).shape == (4, 24, 7, 5)
    classifier = DLinear(config=DLinear.Config(head="Classification", num_classes=3))
    assert classifier(batch_x=batch_x).shape == (4, 3)
    assert classifier.body.linear_trend[0].out_features == 96

    # Reference: Time-Series-Library series_decomp + shared seasonal/trend linears.
    shared = DLinear(config=DLinear.Config())
    front = batch_x[:, :1, :].repeat(1, 12, 1)
    end = batch_x[:, -1:, :].repeat(1, 12, 1)
    padded = torch.cat([front, batch_x, end], dim=1).permute(0, 2, 1)
    trend = functional.avg_pool1d(padded, kernel_size=25, stride=1).permute(0, 2, 1)
    seasonal = batch_x - trend
    body = shared.body
    expected = body.linear_seasonal[0](seasonal.permute(0, 2, 1)) + (
        body.linear_trend[0](trend.permute(0, 2, 1))
    )
    torch.testing.assert_close(shared(batch_x=batch_x), expected.permute(0, 2, 1))
    # Native start; the official 1 / input_length uses Average body-weight patterns.
    weight = shared.state_dict()["body.linear_trend.0.weight"]
    assert not torch.equal(weight, torch.full((24, 96), 1 / 96))


def test_naive_repeats_the_last_observed_step() -> None:
    from kairots.models.Naive import Naive

    batch_x = torch.arange(24.0).reshape(2, 4, 3)
    forecast = Naive(config=Naive.Config(output_length=5))(batch_x=batch_x)
    assert forecast.shape == (2, 5, 3)
    assert torch.equal(forecast, batch_x[:, -1:, :].repeat(1, 5, 1))
    assert not list(Naive(config=Naive.Config()).parameters())


def test_timesnet_task_outputs_match_tslib_contract() -> None:
    import pytest

    reference = Path(__file__).resolve().parents[2] / "tmp" / "Time-Series-Library"
    if not reference.is_dir():
        pytest.skip("Time-Series-Library reference checkout is missing")
    sys.path.insert(0, str(reference))
    try:
        reference_model = importlib.import_module("models.TimesNet").Model
    except ImportError as error:
        pytest.skip(f"Time-Series-Library dependencies are unavailable: {error}")

    from kairots.models.TimesNet import TimesNet

    for head, task_name, input_length, output_length, channels, classes in (
        ("Forecasting", "long_term_forecast", 32, 8, 3, 4),
        ("Forecasting", "long_term_forecast", 96, 96, 3, 4),
        ("Forecasting", "anomaly_detection", 32, 32, 3, 4),
        ("Classification", "classification", 32, 0, 3, 4),
    ):
        torch.manual_seed(0)
        config = SimpleNamespace(
            **{
                "task_name": task_name,
                "seq_len": input_length,
                "label_len": 0,
                "pred_len": 0 if task_name == "anomaly_detection" else output_length,
                "top_k": 2,
                "d_model": 8,
                "d_ff": 16,
                "num_kernels": 3,
                "e_layers": 2,
                "enc_in": channels,
                "c_out": channels,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                "num_class": classes,
            }
        )
        theirs = reference_model(configs=config).eval()
        ours = TimesNet(
            config=TimesNet.Config(
                head=head,
                reconstruct=task_name == "anomaly_detection",
                input_length=input_length,
                output_length=output_length,
                input_channels=channels,
                num_classes=classes,
                d_model=config.d_model,
                d_ff=config.d_ff,
                encoder_layers=config.e_layers,
                k=config.top_k,
                num_inception_kernels=config.num_kernels,
                dropout=config.dropout,
            )
        ).eval()
        batch_x = torch.randn(2, input_length, channels)
        with torch.no_grad():
            if task_name == "classification":
                expected = theirs(batch_x, torch.ones(2, input_length), None, None)
            else:
                expected = theirs(batch_x, None, None, None)
            actual = ours(batch_x=batch_x)
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()


def test_timesnet_period_uses_paper_ceiling_for_nondivisible_frequency() -> None:
    from kairots.models.TimesNet.TimesBlock import TimesBlock

    time_length = 10
    time = torch.arange(time_length, dtype=torch.float32)
    features = torch.sin(2 * torch.pi * 3 * time / time_length)[None, :, None]
    block = TimesBlock(d_model=1, d_ff=2, k=1, num_inception_kernels=2).eval()

    spectrum = torch.fft.rfft(features, dim=1)
    amplitudes = spectrum.abs().mean(dim=0).mean(dim=-1)
    frequency = int(torch.topk(amplitudes[1:], 1).indices[0]) + 1
    assert frequency == 3
    period = (time_length + frequency - 1) // frequency
    padded_length = ((time_length + period - 1) // period) * period
    padded = torch.cat(
        (features, features.new_zeros((1, padded_length - time_length, 1))), dim=1
    )
    arranged = padded.reshape(1, padded_length // period, period, 1).permute(0, 3, 1, 2)
    transformed = block.conv[2](
        features=block.conv[1](block.conv[0](features=arranged))
    )
    restored = transformed.permute(0, 2, 3, 1).reshape(1, -1, 1)[:, :time_length]
    expected = restored + features

    torch.testing.assert_close(block(features=features), expected)


def _timesnet_imputation_pair(
    *, input_length: int, channels: int, marks: int
) -> tuple[Any, Any]:
    import pytest

    reference = Path(__file__).resolve().parents[2] / "tmp" / "Time-Series-Library"
    if not reference.is_dir():
        pytest.skip("Time-Series-Library reference checkout is missing")
    sys.path.insert(0, str(reference))
    try:
        reference_model = importlib.import_module("models.TimesNet").Model
    except ImportError as error:
        pytest.skip(f"Time-Series-Library dependencies are unavailable: {error}")

    from kairots.models.TimesNet import TimesNet

    config = SimpleNamespace(
        task_name="imputation",
        seq_len=input_length,
        label_len=0,
        pred_len=0,
        top_k=2,
        d_model=8,
        d_ff=16,
        num_kernels=3,
        e_layers=2,
        enc_in=channels,
        c_out=channels,
        embed="timeF",
        freq="h",
        dropout=0.1,
        num_class=2,
    )
    torch.manual_seed(0)
    theirs = reference_model(configs=config).eval()
    ours = TimesNet(
        config=TimesNet.Config(
            head="Imputation",
            reconstruct=True,
            input_length=input_length,
            output_length=input_length,
            input_channels=channels,
            calendar_channels=marks,
            d_model=8,
            d_ff=16,
            encoder_layers=2,
            k=2,
            num_inception_kernels=3,
            dropout=0.1,
        )
    ).eval()
    for source, target in zip(theirs.parameters(), ours.parameters(), strict=True):
        assert source.shape == target.shape
        target.data.copy_(source.data)
    return theirs, ours


def test_timesnet_imputation_matches_tslib_with_mask_aware_normalization() -> None:
    theirs, ours = _timesnet_imputation_pair(input_length=24, channels=3, marks=4)
    generator = torch.Generator().manual_seed(1)
    # Hidden positions are far from the observed mean, so statistics over all
    # positions (zeros included) would differ visibly from observed-only ones.
    values = torch.randn(2, 24, 3, generator=generator) + 5.0
    mask = (torch.rand(2, 24, 3, generator=generator) > 0.4).float()
    marks = torch.rand(2, 24, 4, generator=generator) - 0.5
    count = mask.sum(dim=1, keepdim=True)
    means = ((values * mask).sum(dim=1, keepdim=True) / count).detach()
    centered = (values - means).masked_fill(mask == 0, 0)
    stdev = torch.sqrt((centered.square().sum(dim=1, keepdim=True) / count) + 1e-5)
    normalized = centered / stdev
    actual_embeddings: list[torch.Tensor] = []
    hook = ours.body.enc_embedding.register_forward_hook(
        lambda _module, _args, output: actual_embeddings.append(output)
    )
    with torch.no_grad():
        expected = theirs.enc_embedding(normalized, marks)
        actual = ours(batch_x=torch.cat([values * mask, mask, marks], dim=-1))
    hook.remove()
    assert torch.allclose(actual_embeddings[0], expected, atol=1e-5)
    assert actual.shape == values.shape


def test_timesnet_imputation_time_marks_change_the_output_as_in_tslib() -> None:
    theirs, ours = _timesnet_imputation_pair(input_length=24, channels=3, marks=4)
    generator = torch.Generator().manual_seed(2)
    values = torch.randn(2, 24, 3, generator=generator)
    mask = torch.ones(2, 24, 3)
    first = torch.rand(2, 24, 4, generator=generator) - 0.5
    second = first + 0.3
    with torch.no_grad():
        outputs = [
            ours(batch_x=torch.cat([values, mask, marks], dim=-1))
            for marks in (first, second)
        ]
        expected = theirs(values, second, None, None, mask)
    assert not torch.allclose(outputs[0], outputs[1], atol=1e-4)
    assert torch.allclose(outputs[1], expected, atol=1e-5)


def test_timesnet_calendar_needs_the_imputation_head() -> None:
    import pytest

    from kairots.models.TimesNet import TimesNet

    with pytest.raises(ValueError, match="Imputation head"):
        TimesNet.Config(head="Forecasting", calendar_channels=4)


def test_xgboost_fits_and_round_trips_state() -> None:
    import pytest

    pytest.importorskip("xgboost")
    from kairots.models.XGBoost import XGBoost

    generator = torch.Generator().manual_seed(0)
    batch_x = torch.randn(64, 4, 2, generator=generator)
    batch_y = batch_x[:, -2:, :] * 2 + 0.5
    config = XGBoost.Config(
        input_length=4, output_length=2, input_channels=2, num_trees=20, max_depth=2
    )
    model = XGBoost(config=config)
    assert torch.equal(model(batch_x=batch_x[:3]), torch.zeros(3, 2, 2))
    model.fit(batch_x=batch_x, batch_y=batch_y)
    prediction = model(batch_x=batch_x[:3])
    error = torch.mean((prediction - batch_y[:3]) ** 2).item()
    assert error < 0.1 * torch.mean(batch_y[:3] ** 2).item()
    restored = XGBoost(config=config)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(batch_x=batch_x[:3]), prediction)
    size = model.booster.numel()
    model.fit(batch_x=batch_x, batch_y=batch_y)
    assert model.booster.numel() > size
