from __future__ import annotations

import pytest
import torch

from kairots.metrics.CRPSSum import CRPSSum
from kairots.metrics.SampleCRPS import sample_crps
from kairots.models.CSDI import CSDI
from kairots.models.TimeGrad import TimeGrad
from kairots.tasks.ProbabilisticImputation import ProbabilisticImputation


def test_sample_crps_matches_pairwise_and_retains_joint_samples() -> None:
    prediction = torch.tensor([[[[0.0, 2.0, 4.0], [4.0, 2.0, 0.0]]]])
    target = torch.tensor([[[1.0, 3.0]]])
    total, count = sample_crps(prediction=prediction, target=target)
    exact = (prediction - target.unsqueeze(-1)).abs().mean(dim=-1)
    exact -= (prediction.unsqueeze(-1) - prediction.unsqueeze(-2)).abs().mean(
        dim=(-1, -2)
    ) / 2
    assert total.item() == pytest.approx(exact.sum().item())
    assert count.item() == 2
    metric = CRPSSum()
    metric.update(prediction=prediction, target=target)
    assert metric.compute() == 0


def test_csdi_masks_native_missing_values_and_preserves_observed_zeros() -> None:
    task = ProbabilisticImputation(
        config=ProbabilisticImputation.Config(mask_ratio=0.5)
    )
    target = torch.tensor([[[0.0, 1.0], [float("nan"), 3.0], [4.0, 5.0], [6.0, 7.0]]])
    batch_x, batch_y = task.prepare(
        batch_x=target,
        batch_y=target,
        output_length=4,
        generator=torch.Generator().manual_seed(5),
    )
    assert batch_x[..., 2:].sum() + batch_y.isfinite().sum() == target.isfinite().sum()
    assert batch_y.isfinite().sum() == 3  # floor(0.5 * 7), as in CSDI's parser.
    assert batch_x[0, 1, 2] == 0
    model = CSDI(
        config=CSDI.Config(
            input_length=4,
            output_length=4,
            input_channels=2,
            width=8,
            heads=2,
            layers=1,
            diffusion_steps=3,
            samples=4,
        )
    )
    loss = model.training_loss(batch_x=batch_x, target=batch_y)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.output_projection[-1].weight.grad is not None
    model.eval()
    with torch.no_grad():
        samples = model(batch_x=batch_x)
    assert samples.shape == (1, 4, 2, 4)
    mask = batch_x[..., 2:].bool()
    assert torch.equal(samples[mask], batch_x[..., :2][mask, None].expand(-1, 4))
    assert torch.equal(samples, model(batch_x=batch_x))
    assert model.beta.tolist() == pytest.approx(
        torch.linspace(0.0001**0.5, 0.5**0.5, steps=3).square().tolist()
    )


def test_csdi_masks_accept_strided_panel_batches() -> None:
    values = torch.arange(16.0).reshape(1, 2, 8).transpose(1, 2)
    task = ProbabilisticImputation(
        config=ProbabilisticImputation.Config(mask_ratio=0.5)
    )
    features, target = task.prepare(
        batch_x=values,
        batch_y=values,
        output_length=8,
        generator=torch.Generator().manual_seed(1),
    )
    assert target.isfinite().sum() == 8
    model = CSDI(
        config=CSDI.Config(
            input_length=8,
            input_channels=2,
            width=8,
            heads=2,
            layers=1,
            diffusion_steps=3,
            samples=2,
        )
    )
    assert model.training_loss(batch_x=features, target=target).isfinite()


def test_csdi_samples_in_chunks_without_changing_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CSDI(
        config=CSDI.Config(
            input_length=4,
            input_channels=2,
            width=4,
            heads=2,
            layers=1,
            diffusion_steps=2,
            samples=9,
        )
    )
    calls = []
    predict_noise = model.predict_noise

    def track_batch(*, noisy: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        calls.append(len(noisy))
        return predict_noise(noisy=noisy, **kwargs)

    monkeypatch.setattr(model, "predict_noise", track_batch)
    batch_x = torch.randn(17, 4, 4)
    with torch.no_grad():
        output = model(batch_x=batch_x)
        repeated = model(batch_x=batch_x)
    assert output.shape == (17, 4, 2, 9)
    assert torch.equal(output, repeated)
    assert max(calls) <= 128


def test_csdi_historical_and_test_masks_use_patterns_only() -> None:
    model = CSDI(
        config=CSDI.Config(
            input_length=4,
            input_channels=2,
            target_strategy="historical",
            width=8,
            heads=2,
            layers=1,
        )
    )
    observed = torch.tensor([[[True, False]] * 4])
    x = torch.cat([torch.zeros(1, 4, 2), observed.float()], dim=-1)
    model.set_patterns(batches=[(x, torch.full((1, 4, 2), float("nan")))])
    assert torch.equal(
        model.conditional_mask(observed=torch.ones_like(observed)), observed
    )
    model.config.target_strategy = "test"
    model.config.target_pattern = ((True, False),) * 4
    assert torch.equal(
        model.conditional_mask(observed=torch.ones_like(observed)), observed
    )


def test_timegrad_prediction_ignores_future_observations_and_scales_context_mean() -> (
    None
):
    model = TimeGrad(
        config=TimeGrad.Config(
            input_length=6,
            output_length=2,
            input_channels=2,
            output_channels=2,
            calendar_channels=2,
            lags=(1, 2),
            cells=4,
            recurrent_layers=2,
            width=4,
            layers=2,
            diffusion_steps=3,
            samples=4,
        )
    )
    batch_x = torch.arange(1, 33).float().reshape(1, 8, 4)
    history = batch_x[:, :6, :2]
    assert torch.equal(
        model.context_scale(history=history), history[:, -4:].mean(dim=1, keepdim=True)
    )
    assert torch.equal(
        model.context_scale(history=torch.zeros_like(history)), torch.ones(1, 1, 2)
    )
    target = torch.ones(1, 2, 2)
    loss = model.training_loss(batch_x=batch_x, target=target)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.output_projection.weight.grad is not None
    model.eval()
    with torch.no_grad():
        samples = model(batch_x=batch_x)
        batch_x[:, 6:, :2] = float("nan")
        assert torch.equal(samples, model(batch_x=batch_x))
    assert samples.shape == (1, 2, 2, 4)
    assert model.beta.tolist() == pytest.approx([0.0001, 0.05005, 0.1])


def test_timegrad_lags_are_strictly_causal_and_channel_major() -> None:
    model = TimeGrad(
        config=TimeGrad.Config(
            input_length=6,
            output_length=2,
            input_channels=2,
            output_channels=2,
            calendar_channels=0,
            lags=(1, 2),
        )
    )
    sequence = torch.arange(16).reshape(1, 8, 2).float()
    features = model.recurrent_features(
        sequence=sequence, calendar=torch.empty(1, 8, 0), start=2, end=4
    )
    assert torch.equal(
        features[..., :4], torch.tensor([[[2.0, 0.0, 3.0, 1.0], [4.0, 2.0, 5.0, 3.0]]])
    )


def test_csdi_timestamp_covariate_and_known_pattern() -> None:
    task = ProbabilisticImputation(
        config=ProbabilisticImputation.Config(
            input_channels=("a", "b"),
            calendar=("ElapsedHours",),
            target_pattern=((True, False),) * 4,
        )
    )
    values = torch.tensor(
        [[[0.0, 1.0, 0.0], [2.0, 3.0, 0.5], [4.0, 5.0, 2.0], [6.0, 7.0, 4.0]]]
    )
    batch_x, target = task.prepare(
        batch_x=values,
        batch_y=values,
        output_length=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(batch_x[..., -1], values[..., -1])
    assert target[..., 0].isnan().all()
    assert target[..., 1].isfinite().all()
    model = CSDI(
        config=CSDI.Config(
            input_length=4,
            input_channels=2,
            width=8,
            heads=2,
            layers=1,
            diffusion_steps=3,
            samples=2,
        )
    )
    assert torch.isfinite(model.training_loss(batch_x=batch_x, target=target))


def test_csdi_reverse_chain_matches_ddpm_equations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CSDI(
        config=CSDI.Config(
            input_length=4,
            input_channels=2,
            width=8,
            heads=2,
            layers=1,
            diffusion_steps=3,
            samples=2,
        )
    )

    def zero_noise(*, noisy: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return torch.zeros_like(noisy)

    monkeypatch.setattr(model, "predict_noise", zero_noise)
    batch_x = torch.zeros(1, 4, 4)
    actual = model(batch_x=batch_x)
    generator = torch.Generator().manual_seed(model.config.seed)
    expected = torch.randn(2, 4, 2, generator=generator)
    beta = torch.linspace(0.0001**0.5, 0.5**0.5, steps=3).square()
    alpha = (1 - beta).cumprod(dim=0)
    for index in (2, 1, 0):
        expected = expected / (1 - beta[index]).sqrt()
        if index:
            variance = beta[index] * (1 - alpha[index - 1]) / (1 - alpha[index])
            expected += variance.sqrt() * torch.randn(
                expected.shape, generator=generator
            )
    assert torch.equal(actual, expected.reshape(1, 2, 4, 2).permute(0, 2, 3, 1))


@pytest.mark.parametrize("name", ["CSDI", "TimeGrad"])
def test_diffusion_objective_sums_vector_coordinates(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def zero_noise(*, noisy: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return torch.zeros_like(noisy)

    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))
    if name == "CSDI":
        model = CSDI(
            config=CSDI.Config(
                input_length=4,
                input_channels=2,
                width=8,
                heads=2,
                layers=1,
                target_strategy="test",
                target_pattern=((True, False),) * 4,
            )
        )
        batch_x = torch.cat([torch.ones(2, 4, 2), torch.zeros(2, 4, 2)], dim=-1)
        target = torch.ones(2, 4, 2)
        expected = 4
    else:
        model = TimeGrad(
            config=TimeGrad.Config(
                input_length=6,
                output_length=2,
                input_channels=2,
                output_channels=2,
                calendar_channels=0,
                lags=(1, 2),
            )
        )
        batch_x = torch.ones(2, 8, 2)
        target = torch.ones(2, 2, 2)
        expected = 2
    monkeypatch.setattr(model, "predict_noise", zero_noise)
    assert model.training_loss(batch_x=batch_x, target=target).item() == expected


def test_csdi_quantile_crps_and_median_match_definitions() -> None:
    from kairots.metrics.MedianMAE import MedianMAE
    from kairots.metrics.NormalizedCRPS import NormalizedCRPS

    samples = torch.tensor([[[[0.0, 2.0, 4.0, 6.0], [999.0, 999.0, 999.0, 999.0]]]])
    target = torch.tensor([[[2.0, float("inf")]]])
    metric = NormalizedCRPS()
    metric.update(prediction=samples, target=target)
    levels = torch.arange(1, 20).float() / 20
    quantiles = 6 * levels
    error = 2 - quantiles
    expected = 2 * torch.maximum(levels * error, (levels - 1) * error).mean() / 2
    assert metric.compute() == pytest.approx(expected.item())
    median = MedianMAE()
    median.update(prediction=samples, target=target)
    assert median.compute() == 0


def test_multistep_matches_paper_epoch_milestones_and_resumes() -> None:
    from kairots.schedulers.MultiStep import MultiStep

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    scheduler = MultiStep(config=MultiStep.Config()).build(
        optimizer=optimizer, iterations=200
    )
    rates = []
    for index in range(200):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[149] == 0.001
    assert rates[150] == pytest.approx(0.0001)
    assert rates[180] == pytest.approx(0.00001)
    resumed = MultiStep(config=MultiStep.Config()).build(
        optimizer=optimizer, iterations=200
    )
    resumed.load_state_dict(scheduler.state_dict())
    assert resumed.state_dict() == scheduler.state_dict()


def test_diffusion_paper_architecture_defaults_and_prior_calendar_alignment() -> None:
    csdi = CSDI(config=CSDI.Config(input_channels=35))
    assert 410000 <= sum(value.numel() for value in csdi.parameters()) <= 420000
    assert len(csdi.blocks) == 4
    encoder = dict(csdi.named_modules())["blocks.0.temporal.layers.0"]
    assert isinstance(encoder, torch.nn.TransformerEncoderLayer)
    assert encoder.linear1.out_features == 64
    assert encoder.activation.__name__ == "gelu"
    assert encoder.dropout.p == 0.1
    timegrad = TimeGrad(config=TimeGrad.Config())
    assert timegrad.recurrent.num_layers == 2
    assert timegrad.recurrent.weight_hh_l0.shape == (160, 40)
    assert len(timegrad.blocks) == 8
    calendar = torch.arange(50).float().reshape(1, 50, 1).expand(1, 50, 4)
    features = timegrad.recurrent_features(
        sequence=torch.ones(1, 50, 8), calendar=calendar, start=14, end=16
    )
    assert torch.equal(features[..., -4:], calendar[:, 13:15])


def test_csdi_forward_noising_matches_resolved_ddpm_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CSDI(
        config=CSDI.Config(
            input_length=4,
            input_channels=2,
            width=8,
            heads=2,
            layers=1,
            diffusion_steps=3,
            target_strategy="test",
            target_pattern=((False, False),) * 4,
        )
    )
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))

    def fixed_step(
        *args: object, size: tuple[int, ...], **kwargs: object
    ) -> torch.Tensor:
        return torch.ones(size, dtype=torch.long)

    monkeypatch.setattr(torch, "randint", fixed_step)

    def identity_noise(*, noisy: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return noisy

    monkeypatch.setattr(model, "predict_noise", identity_noise)
    batch_x = torch.zeros(1, 4, 4)
    target = torch.full((1, 4, 2), 2.0)
    alpha = model.alpha[1]
    expected = (2 * alpha.sqrt() + (1 - alpha).sqrt() - 1).square() * 8
    assert model.training_loss(batch_x=batch_x, target=target).item() == pytest.approx(
        expected.item()
    )


def test_diffusion_learner_and_task_contracts_reject_incompatible_configs() -> None:
    from kairots.config.schema import Experiment

    assert (
        Experiment.from_dict(
            values={
                "task": {"name": "ProbabilisticImputation"},
                "model": {"name": "CSDI"},
            }
        ).learner.name
        == "Denoising"
    )
    with pytest.raises(ValueError, match="learner.name"):
        Experiment.from_dict(
            values={
                "task": {"name": "ProbabilisticImputation"},
                "model": {"name": "CSDI"},
                "learner": {"name": "Gradient"},
            }
        )
    with pytest.raises(ValueError, match="learner.precision"):
        Experiment.from_dict(
            values={
                "task": {"name": "SampleForecasting"},
                "model": {"name": "TimeGrad"},
                "learner": {"name": "Denoising", "precision": "bf16"},
            }
        )
    with pytest.raises(ValueError, match="model.name"):
        Experiment.from_dict(values={"learner": {"name": "Denoising"}})
    with pytest.raises(ValueError, match="does not support task"):
        Experiment.from_dict(
            values={"model": {"name": "TimeGrad"}, "learner": {"name": "Denoising"}}
        )
