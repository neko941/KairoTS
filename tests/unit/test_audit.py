"""Regression cases for configuration, numerical and download boundary failures."""

import hashlib
from pathlib import Path

import pytest
import torch

from kairots.__main__ import overlay
from kairots.analysis.report import summarize
from kairots.config.schema import Experiment
from kairots.data.download import RemoteFile, fetch
from kairots.metrics.R2 import R2
from kairots.models.TimesNet import TimesNet
from kairots.training.Online import Online


def test_cli_validates_complete_override_set(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    actual = overlay(
        path=str(config),
        arguments=["--strategy.name", "FedBABU", "--evaluation.finetune_epochs", "1"],
    )
    expected = Experiment.from_dict(
        values={"strategy": {"name": "FedBABU"}, "evaluation": {"finetune_epochs": 1}}
    )
    assert actual.to_dict() == expected.to_dict()


def test_report_counts_finite_metric_values() -> None:
    identity = dict(
        task="AnomalyDetection",
        source="SMD",
        model="Linear",
        strategy="FedAvg",
        config="a",
        status="completed",
    )
    rows = [{**identity, "f1": value} for value in (float("nan"), 0.5)]
    (result,) = summarize(rows=rows, metrics=["f1", "missing"])
    assert result["f1_mean"] == 0.5 and result["f1_std"] == 0
    assert result["f1_n"] == 1
    assert result["missing_mean"] is None and result["missing_n"] == 0


def test_report_ignores_non_numeric_metric_values() -> None:
    identity = dict(
        task="Forecasting",
        source="ETTh1",
        model="Linear",
        strategy="FedAvg",
        config="a",
        status="completed",
    )
    rows = [{**identity, "mse": value} for value in (None, "pending", True, 0.5)]
    (result,) = summarize(rows=rows, metrics=["mse"])
    assert result["mse_n"] == 1
    assert result["mse_mean"] == 0.5 and result["mse_std"] == 0


def test_r2_ignores_unscored_channels_and_marks_constant_targets_undefined() -> None:
    metric = R2()
    metric.update(
        prediction=torch.ones(1, 2, 2),
        target=torch.tensor([[[1.0, float("nan")], [2.0, float("nan")]]]),
    )
    assert metric.compute() == pytest.approx(-1)
    empty = R2()
    empty.update(
        prediction=torch.ones(1, 2, 1), target=torch.full((1, 2, 1), float("nan"))
    )
    with pytest.raises(ValueError, match="no observations"):
        empty.compute()
    constant = R2()
    constant.update(prediction=torch.zeros(1, 2, 1), target=torch.ones(1, 2, 1))
    assert torch.isnan(torch.tensor(constant.compute()))


def test_checksum_failure_uses_next_mirror(tmp_path: Path) -> None:
    bad, good = tmp_path / "bad", tmp_path / "good"
    bad.write_bytes(b"bad")
    good.write_bytes(b"good")
    root = tmp_path / "download"
    result = fetch(
        remote=RemoteFile(
            path="data",
            urls=(bad.as_uri(), good.as_uri()),
            sha256=hashlib.sha256(b"good").hexdigest(),
        ),
        root=root,
    )
    assert result.read_bytes() == b"good"
    assert not list(root.glob("*.part")) and not list(root.glob(".*.part"))


def test_timesnet_excludes_dc_and_validates_available_periods() -> None:
    def model(*, k: int) -> TimesNet:
        return TimesNet(
            config=TimesNet.Config(
                head="Classification",
                input_length=8,
                input_channels=2,
                num_classes=2,
                d_model=4,
                d_ff=8,
                encoder_layers=1,
                k=k,
            )
        )

    with pytest.raises(ValueError, match="k.*8"):
        model(k=5)(batch_x=torch.zeros(2, 8, 2))
    assert model(k=4)(batch_x=torch.zeros(2, 8, 2)).shape == (2, 2)


def test_replay_owns_each_admitted_tensor_storage() -> None:
    batch_x, batch_y = torch.ones(64, 100, 8), torch.ones(64, 3, 8)
    regime = Online(config=Online.Config(steps=0, replay_size=1, capacity=1))
    regime.observe(batch_x=batch_x, batch_y=batch_y, ends=[3] * 64)
    regime.release(model=torch.nn.Linear(1, 1), observed=3, loss_function=None)
    assert regime.replay is not None
    for value in regime.replay.items[0][:2]:
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    batch_x.fill_(999)
    batch_y.fill_(999)
    assert all(
        torch.equal(value, torch.ones_like(value))
        for value in regime.replay.items[0][:2]
    )
