"""Missing data must be explicitly accepted before split filtering or fitting."""

import json
from pathlib import Path

import polars as pl
import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources
from integration.test_classification import _config, _raw

from kairots.config.schema import Experiment
from kairots.data.stores.Parquet import Parquet
from kairots.scalers.Standard import Standard


@pytest.mark.parametrize("imputation", ["zero", "forward", "", False])
def test_standard_rejects_unknown_imputation(imputation: str) -> None:
    with pytest.raises(ValueError, match="data.scaler.imputation"):
        Standard.Config(imputation=imputation)


def test_standard_requires_explicit_masked_handling() -> None:
    values = torch.tensor([[1.0, float("nan")]])
    with pytest.raises(ValueError, match="data.scaler.imputation"):
        Standard.fit(values=values, config=Standard.Config())


def test_masked_and_default_statistics_are_exact_on_complete_data() -> None:
    values = torch.tensor([[1.0, 5.0], [3.0, 5.0], [9.0, 5.0]])
    ordinary = Standard.fit(values=values, config=Standard.Config())
    masked = Standard.fit(values=values, config=Standard.Config(imputation="masked"))
    assert torch.equal(ordinary.mean, masked.mean)
    assert torch.equal(ordinary.scale, masked.scale)


@pytest.mark.parametrize("position", [0, 160, 239])
@pytest.mark.parametrize("missing", [None, float("nan")])
def test_series_missing_values_reject_before_split_filtering(
    tmp_path: Path, position: int, missing: float | None
) -> None:
    raw = tmp_path / "raw"
    write_ett_sources(path=raw)
    path = raw / "ETTh1" / "ETTh1.csv"
    frame = pl.read_csv(path).with_row_index("row")
    frame.with_columns(
        pl.when(pl.col("row") == position)
        .then(missing)
        .otherwise(pl.col("OT"))
        .alias("OT")
    ).drop("row").write_csv(path)
    experiment = ett_experiment(raw_path=raw, runs_dir=tmp_path / "runs")
    with pytest.raises(
        ValueError, match=r"ETDatasetHour.*ETTh1.*OT.*data.scaler.imputation"
    ):
        Parquet(config=experiment.data.store).prepare(experiment=experiment)


@pytest.mark.parametrize("fold", ["TRAIN", "TEST"])
def test_panel_missing_values_reject_in_train_and_test(
    tmp_path: Path, fold: str
) -> None:
    raw = _raw(root=tmp_path / "raw")
    path = raw / "BasicMotions" / f"BasicMotions_{fold}.ts"
    header, data = path.read_text().split("@data\n")
    data = "nan," + data.split(",", 1)[1]
    path.write_text(header + "@data\n" + data)
    values = _config(root=tmp_path, raw=raw)
    experiment = Experiment.from_dict(values=values)
    with pytest.raises(ValueError, match=r"BasicMotions.*data.scaler.imputation"):
        Parquet(config=experiment.data.store).prepare(experiment=experiment)
    values["data"]["scaler"]["imputation"] = "masked"
    experiment = Experiment.from_dict(values=values)
    _, records = Parquet(config=experiment.data.store).prepare(experiment=experiment)
    assert records
    assert any(Parquet.load(record=record)[0].isnan().any() for record in records)


def test_irregular_source_fills_masked_and_rejects_explicit_none() -> None:
    values = json.loads(
        (
            Path(__file__).parents[2]
            / "configs/examples/classification/PhysioNet2012.json"
        ).read_text()
    )
    assert Experiment.from_dict(values=values).data.scaler.imputation == "masked"
    values["data"]["scaler"]["imputation"] = None
    with pytest.raises(ValueError, match="data.scaler.imputation"):
        Experiment.from_dict(values=values)


def test_single_channel_store_uses_writable_arrays(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_ett_sources(path=raw)
    experiment = Experiment.from_dict(
        values={
            "data": {
                "source": {
                    "raw_path": str(raw),
                    "download": False,
                    "input_channels": ["OT"],
                },
                "split": {"input_length": 8, "output_length": 3, "min_windows": 1},
                "store": {"root": str(tmp_path / "datasets")},
            }
        }
    )
    _, records = Parquet(config=experiment.data.store).prepare(experiment=experiment)
    assert records
    for record in records:
        assert torch.isfinite(Parquet.load(record=record)[0]).all()
