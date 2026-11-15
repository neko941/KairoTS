from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from fixtures.synthetic import write_ett_sources
from torch import Tensor, nn

from kairots.config.schema import Experiment
from kairots.data.stores.Parquet import Parquet
from kairots.evaluation.TAFAS import TAFAS, Calibration
from kairots.models.DLinear import DLinear
from kairots.registry import get

INPUT_LENGTH, OUTPUT_LENGTH = 16, 6


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": {"name": "Classification"}},
        {"task": {"calendar": ["HourOfDay"]}},
        {"task": {"target_channels": ["OT"]}},
        {"groups": [{}]},
        {"client_model": {"name": "Linear"}},
    ],
)
def test_adapter_applicability_is_checked_before_running(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="TAFAS"):
        Experiment.from_dict(values={"evaluation": {"name": "TAFAS"}, **overrides})


def _setup(*, workspace_tmp_path: Path, steps: int) -> tuple[Any, Any, dict[str, Any]]:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    experiment = Experiment.from_dict(
        values={
            "data": {
                "split": {
                    "input_length": 8,
                    "output_length": 3,
                    "ratios": [0.6, 0.2, 0.2],
                },
                "source": {
                    "name": "ETDatasetHour",
                    "raw_path": str(raw),
                    "download": False,
                    "batch_size": 4,
                },
                "view": {"name": "Ordered"},
                "store": {"root": str(workspace_tmp_path / "datasets")},
            },
            "evaluation": {"name": "TAFAS", "steps": steps, "learning_rate": 0.01},
            "runtime": {"device": "cpu", "runs_dir": str(workspace_tmp_path / "runs")},
        }
    )
    torch.manual_seed(0)
    model = get(category="models", name=experiment.model.name)(config=experiment.model)
    _, records = Parquet(config=experiment.data.store).prepare(experiment=experiment)
    loss_type = get(category="metrics", name=experiment.evaluation.loss)
    protocol = get(category="evaluation", name="TAFAS")(
        config=experiment.evaluation, loss_function=loss_type(config=loss_type.Config())
    )
    arguments = {
        "phase": "test",
        "model": model,
        "clients": [Parquet.load(record=record) for record in records],
        "experiment": experiment,
        "view": get(category="data/views", name="Ordered")(config=experiment.data.view),
        "task": get(category="tasks", name="Forecasting")(config=experiment.task),
        "device": torch.device("cpu"),
    }
    return model, protocol, arguments


def _model(*, channels: int) -> nn.Module:
    torch.manual_seed(0)
    return DLinear(
        config=DLinear.Config(
            input_length=INPUT_LENGTH,
            output_length=OUTPUT_LENGTH,
            input_channels=channels,
        )
    ).eval()


def _windows(*, series: Tensor) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    count = len(series) - INPUT_LENGTH - OUTPUT_LENGTH + 1
    windows = [(start, start + INPUT_LENGTH) for start in range(count)]
    batch_x = torch.stack(
        [series[start : start + INPUT_LENGTH] for start, _ in windows]
    )
    batch_y = torch.stack(
        [series[target : target + OUTPUT_LENGTH] for _, target in windows]
    )
    return batch_x, batch_y, windows


def _stream(
    *, series: Tensor, steps: int = 1
) -> tuple[Tensor, list[int], list[tuple[int, int]], int]:
    """Walk a DLinear (fixed random weights) over every stride-1 window of `series`."""
    batch_x, batch_y, windows = _windows(series=series)
    protocol = TAFAS(
        config=TAFAS.Config(steps=steps, learning_rate=0.05), loss_function=None
    )
    prediction, final_times = protocol.stream(
        model=_model(channels=series.shape[1]),
        batch_x=batch_x,
        batch_y=batch_y,
        windows=windows,
    )
    return prediction, final_times, windows, protocol.updates


def _series() -> Tensor:
    generator = torch.Generator().manual_seed(1)
    time = torch.arange(240, dtype=torch.float32)[:, None]
    return torch.sin(time * torch.tensor([1.5, 1.2])) + 0.1 * torch.randn(
        240, 2, generator=generator
    )


def test_calibration_is_the_identity_at_initialization() -> None:
    module = Calibration(length=INPUT_LENGTH, channels=3, gating_init=0.01)
    batch = torch.randn(5, INPUT_LENGTH, 3)
    assert torch.equal(module(batch=batch), batch)
    assert (torch.tanh(module.gating) != 0).all()


def test_calibration_matches_variable_wise_affine_maps() -> None:
    module = Calibration(length=5, channels=3, gating_init=0.01).double()
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        module.weight.copy_(torch.randn(5, 5, 3, generator=generator))
        module.bias.copy_(torch.randn(5, 3, generator=generator))
        module.gating.copy_(torch.tensor([-0.5, 0.3, 1.2]))
    batch = torch.randn(2, 5, 3, dtype=torch.float64, generator=generator)
    expected = torch.stack(
        [
            batch[..., channel]
            + torch.tanh(module.gating[channel])
            * (
                batch[..., channel] @ module.weight[..., channel]
                + module.bias[:, channel]
            )
            for channel in range(3)
        ],
        dim=-1,
    )
    torch.testing.assert_close(module(batch=batch), expected, rtol=0, atol=1e-12)


def test_period_uses_the_highest_power_variable() -> None:
    time = torch.arange(96, dtype=torch.float64)
    window = torch.stack(
        [
            torch.sin(2 * torch.pi * time * 4 / 96),
            100 + 4 * torch.sin(2 * torch.pi * time * 13 / 96),
        ],
        dim=-1,
    )
    assert TAFAS.period(window=window) == 8


@pytest.mark.parametrize("frequency,expected", [(4, 24), (13, 8)])
def test_period_is_the_dominant_cycle_length(frequency: int, expected: int) -> None:
    time = torch.arange(96, dtype=torch.float32)
    window = torch.stack(
        [torch.sin(2 * torch.pi * time * frequency / 96), time * 0.0], dim=1
    )
    assert TAFAS.period(window=window) == expected
    assert TAFAS.period(window=torch.zeros(96, 2)) == 24


def test_zero_steps_reproduce_frozen_scores(workspace_tmp_path: Path) -> None:
    _, protocol, arguments = _setup(workspace_tmp_path=workspace_tmp_path, steps=0)
    metrics, _ = protocol.evaluate(**arguments)
    assert metrics["adaptation_updates"] == 0.0
    assert metrics["pooled_mse"] == metrics["frozen_pooled_mse"]


def test_adaptation_changes_scores_and_restores_the_source_model(
    workspace_tmp_path: Path,
) -> None:
    model, protocol, arguments = _setup(workspace_tmp_path=workspace_tmp_path, steps=1)
    source = {key: value.clone() for key, value in model.state_dict().items()}
    metrics, _ = protocol.evaluate(**arguments)
    assert metrics["adaptation_updates"] > 0
    assert metrics["pooled_mse"] != metrics["frozen_pooled_mse"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, source[key]), key
    # The backbone never receives a gradient or loses a parameter flag.
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_unobserved_targets_never_reach_an_earlier_prediction() -> None:
    series = _series()
    clean, clean_times, windows, _ = _stream(series=series)
    # Poison from each time step at which a prediction became final: the first
    # row that was still unobserved then.
    finals = sorted(set(clean_times))
    for cut in [*finals[::3], finals[-1], len(series) - 3]:
        poisoned = series.clone()
        poisoned[cut:] += 5.0
        other, other_times, _, _ = _stream(series=poisoned)
        final = [index for index, time in enumerate(clean_times) if time <= cut]
        assert final, cut
        assert other_times[: len(final)] == clean_times[: len(final)]
        assert torch.equal(other[: len(final)], clean[: len(final)]), cut
        # Where the poison reaches an input window, it does change what comes after.
        if cut < len(series) - OUTPUT_LENGTH:
            assert not torch.equal(other, clean), cut
    assert len(windows) == len(series) - INPUT_LENGTH - OUTPUT_LENGTH + 1


def test_first_mini_batch_matches_a_manual_derivation() -> None:
    """One partial-target step on the first window, then the adjustment."""
    series = _series()
    batch_x, batch_y, _ = _windows(series=series)
    prediction, _, _, _ = _stream(series=series)
    period = TAFAS.period(window=batch_x[0])
    model = _model(channels=series.shape[1])
    modules = [
        Calibration(length=length, channels=series.shape[1], gating_init=0.01)
        for length in (INPUT_LENGTH, OUTPUT_LENGTH)
    ]
    parameters = [item for module in modules for item in module.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=0.05, weight_decay=0.0)
    window = batch_x[: period + 1]

    def forward() -> Tensor:
        return modules[1](batch=model(batch_x=modules[0](batch=window)))

    before = forward()
    loss = (before[0, :period] - batch_y[0, :period]).pow(2).mean()
    for parameter, gradient in zip(
        parameters, torch.autograd.grad(loss, parameters), strict=True
    ):
        parameter.grad = gradient
    optimizer.step()
    expected, adapted = before.detach().clone(), forward().detach()
    for offset in range(period + 1):
        expected[offset, period - offset :] = adapted[offset, period - offset :]
    assert torch.equal(prediction[: period + 1], expected)
    assert not torch.equal(expected, before.detach())
    assert not torch.equal(expected[-1], before.detach()[-1])


def test_full_targets_do_not_add_optimizer_steps() -> None:
    series = _series()
    _, final_times, windows, updates = _stream(series=series)
    # A mini-batch is the run of windows that share a final time.
    batches = []
    for index, time in enumerate(final_times):
        if not batches or batches[-1][0] != time:
            batches.append([time, index, index])
        batches[-1][2] = index
    partial = len(batches) - 1  # the stream's last, cut-short mini-batch has none
    full = sum(
        any(
            later[0] >= windows[last][1] + OUTPUT_LENGTH
            for later in batches[position + 1 :]
        )
        for position, (_, _, last) in enumerate(batches)
    )
    assert full > 0
    assert updates == partial


def test_joint_loss_uses_the_most_recent_available_full_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # At the third update two old batches mature; Eq5 chooses the newer one.
    periods = iter((2, 2, 5))
    monkeypatch.setattr(TAFAS, "period", staticmethod(lambda **_: next(periods)))
    series = _series()[: INPUT_LENGTH + OUTPUT_LENGTH + 11]
    batch_x, batch_y, _ = _windows(series=series)
    prediction, _, _, updates = _stream(series=series)
    assert updates == 3
    model = _model(channels=series.shape[1])
    modules = [
        Calibration(length=length, channels=series.shape[1], gating_init=0.01)
        for length in (INPUT_LENGTH, OUTPUT_LENGTH)
    ]
    parameters = [item for module in modules for item in module.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=0.05, weight_decay=0.0)

    def forecast(batch: Tensor) -> Tensor:
        return modules[1](batch=model(batch_x=modules[0](batch=batch)))

    for first, last, period in ((0, 3, 2), (3, 6, 2), (6, 12, 5)):
        optimizer.zero_grad(set_to_none=True)
        original = forecast(batch_x[first:last])
        loss = (original[0, :period] - batch_y[first, :period]).pow(2).mean()
        if first == 6:
            loss = loss + (forecast(batch_x[3:6]) - batch_y[3:6]).pow(2).mean()
        loss.backward(inputs=parameters)
        optimizer.step()
    expected = original.detach().clone()
    adapted = forecast(batch_x[6:12]).detach()
    for offset in range(6):
        expected[offset, 5 - offset :] = adapted[offset, 5 - offset :]
    torch.testing.assert_close(prediction[6:12], expected, rtol=0, atol=0)


def test_non_stride_one_stream_is_rejected() -> None:
    series = _series()
    protocol = TAFAS(config=TAFAS.Config(), loss_function=None)
    windows = [(0, INPUT_LENGTH), (2, INPUT_LENGTH + 2)]
    batch_x = torch.stack(
        [series[start : start + INPUT_LENGTH] for start, _ in windows]
    )
    batch_y = torch.stack(
        [series[target : target + OUTPUT_LENGTH] for _, target in windows]
    )
    with pytest.raises(ValueError, match="stride 1"):
        protocol.stream(
            model=_model(channels=2),
            batch_x=batch_x,
            batch_y=batch_y,
            windows=windows,
        )
