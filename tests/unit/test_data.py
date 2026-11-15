from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from kairots.data.splits.CompleteWindowRatio import CompleteWindowRatio
from kairots.scalers.Identity import Identity


def test_identity_scaler_returns_values_unchanged() -> None:
    values = torch.tensor([[1.0, 4.0], [3.0, 4.0]])
    scaler = Identity.fit(values=values, config=Identity.Config())
    assert scaler.mean.tolist() == [0.0, 0.0]
    assert scaler.scale.tolist() == [1.0, 1.0]


def _frame(*, count: int, gap: int | None = None) -> pl.DataFrame:
    dates = [
        index for index in range(count + (1 if gap is not None else 0)) if index != gap
    ]
    return pl.DataFrame({"date": dates, "v": np.arange(len(dates), dtype=np.float32)})


@pytest.mark.parametrize("gap", [None, 15])
def test_complete_window_ratio_target_ranges_are_disjoint_and_cover(
    *, gap: int | None
) -> None:
    frame = _frame(count=80, gap=gap).with_columns(pl.col("date").cast(pl.Int64))
    splits = CompleteWindowRatio().split(
        frame=frame,
        input_length=5,
        output_length=3,
        offset_length=0,
        ratios=(0.6, 0.2, 0.2),
        channels=("v",),
        lookback_crosses_boundary=True,
        min_windows=2,
    )
    target_sets = [
        {
            int(frame["date"][position])
            for _, target in splits[name]
            for position in range(target, target + 3)
        }
        for name in ("train", "validation", "test")
    ]
    assert all(
        not target_sets[left] & target_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    )


def test_complete_window_ratio_skips_gaps_and_nans_without_a_three_split_floor() -> (
    None
):
    frame = _frame(count=26).with_columns(pl.col("date").cast(pl.Int64))
    frame = frame.with_columns(
        pl.when(pl.col("date") == 3).then(None).otherwise(pl.col("v")).alias("v")
    )
    splits = CompleteWindowRatio().split(
        frame=frame,
        input_length=1,
        output_length=1,
        offset_length=0,
        ratios=(0.5, 0.0, 0.5),
        channels=("v",),
        lookback_crosses_boundary=True,
        min_windows=10,
    )
    # 25 windows of two rows, minus the two that touch the missing row 3.
    assert len(splits["train"]) + len(splits["test"]) == 23
    assert not splits["validation"]
    assert all(start not in (2, 3) for start, _ in splits["train"] + splits["test"])
    gapped = _frame(count=20, gap=10).with_columns(pl.col("date").cast(pl.Int64))
    assert CompleteWindowRatio._starts(
        frame=gapped, channels=("v",), length=3
    ).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]


def test_shared_calendar_has_one_cut_for_all_clients() -> None:
    frames = [_frame(count=80), _frame(count=90)]
    cuts = CompleteWindowRatio().shared_cut_dates(
        frames=frames,
        input_length=5,
        output_length=1,
        offset_length=0,
        ratios=(0.6, 0.2, 0.2),
        channels=("v",),
    )
    assert all(cut in frame["date"].to_list() for cut in cuts for frame in frames)


def test_fetch_is_atomic_and_checks_sha256(tmp_path: Path) -> None:
    import hashlib

    from kairots.data.download import RemoteFile, fetch
    from kairots.data.stores.Parquet import ranges

    payload = b"date,OT\n" * 1000
    source = tmp_path / "remote.csv"
    source.write_bytes(payload)
    missing = (tmp_path / "nowhere.csv").as_uri()
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="sha256"):
        fetch(
            remote=RemoteFile(path="bad.csv", urls=(source.as_uri(),), sha256="0" * 64),
            root=tmp_path / "raw",
        )
    assert list((tmp_path / "raw").iterdir()) == []
    # A half-written .part resumes, and a dead mirror falls through to the next one.
    (tmp_path / "raw" / ".good.csv.part").write_bytes(payload[:100])
    good = fetch(
        remote=RemoteFile(
            path="good.csv", urls=(missing, source.as_uri()), sha256=digest
        ),
        root=tmp_path / "raw",
    )
    assert good.read_bytes() == payload
    assert sorted(path.name for path in (tmp_path / "raw").iterdir()) == ["good.csv"]
    assert ranges(starts=[1, 2, 3, 7, 8, 10]) == [[1, 3], [7, 8], [10, 10]]


def test_jitter_adds_seeded_noise_of_scale_sigma() -> None:
    from kairots.augmentations.Jitter import Jitter

    jitter = Jitter(config=Jitter.Config(sigma=0.5))
    batch_x = torch.zeros(64, 32, 4)
    first = jitter.apply(batch_x=batch_x, generator=torch.Generator().manual_seed(1))
    again = jitter.apply(batch_x=batch_x, generator=torch.Generator().manual_seed(1))
    assert torch.equal(first, again)
    assert abs(first.std().item() - 0.5) < 0.02


def test_ordered_view_never_shuffles() -> None:
    from kairots.data.views.Offline import Offline
    from kairots.data.views.Ordered import Ordered

    values = torch.arange(40.0).reshape(20, 2)
    windows = [(start, start + 4) for start in range(12)]
    options = {
        "values": values,
        "windows": windows,
        "input_length": 4,
        "output_length": 2,
        "offset_length": 0,
        "batch_size": 5,
        "generator": torch.Generator().manual_seed(0),
    }
    ordered = Ordered(config=Ordered.Config()).batches(**options, shuffle=True)
    expected = Offline(config=Offline.Config()).batches(**options, shuffle=False)
    assert all(
        torch.equal(left[0], right[0]) and torch.equal(left[1], right[1])
        for left, right in zip(ordered, expected, strict=True)
    )


def test_panel_view_returns_sequence_targets_for_reconstruction() -> None:
    from kairots.data.views.Offline import Offline

    values = torch.arange(24.0).reshape(2, 4, 3)
    batches = list(
        Offline(config=Offline.Config()).batches(
            values=values,
            windows=[(0, 0), (1, 0)],
            input_length=0,
            output_length=4,
            offset_length=0,
            batch_size=2,
            shuffle=False,
            generator=torch.Generator().manual_seed(0),
        )
    )
    batch_x, batch_y = batches[0]
    assert torch.equal(batch_x, values)
    assert torch.equal(batch_y, values)


def test_grouped_holds_out_a_seeded_share_of_clients() -> None:
    from kairots.data.splits.Grouped import Grouped

    def unseen(*, ratio: float, count: int) -> list[int]:
        split = Grouped(config=Grouped.Config(unseen_ratio=ratio, unseen_seed=3))
        return split.unseen(client_ids=list(range(count)))

    assert len(unseen(ratio=0.25, count=8)) == 2
    assert unseen(ratio=0.25, count=8) == unseen(ratio=0.25, count=8)
    assert unseen(ratio=0.0, count=8) == []
    assert len(unseen(ratio=0.9, count=2)) == 1


def test_test_fold_forecasts_the_official_fold_from_earlier_history() -> None:
    from kairots.data.splits.TestFold import TestFold

    frame = pl.DataFrame({"date": range(20), "fold": ["train"] * 16 + ["test"] * 4})
    splits = TestFold(
        config=TestFold.Config(
            input_length=4, output_length=2, validation_windows=1, train_stride=1
        )
    ).series(frame=frame, input_length=4, output_length=2, offset_length=0)
    assert splits["test"] == [(12, 16), (14, 18)]
    assert splits["validation"] == [(10, 14)]
    assert splits["train"] and all(target + 2 <= 14 for _, target in splits["train"])
    assert splits["train"][-1] == (8, 12)


def test_per_node_partitions_sorted_sensor_series() -> None:
    from types import SimpleNamespace

    from kairots.data.partitions.PerNode import PerNode

    frame = pl.DataFrame(
        {"date": [2, 1, 2, 1], "node": [0, 0, 1, 1], "v": [2, 1, 4, 3]}
    )
    source = SimpleNamespace(clients=[("Net", frame)])
    clients = PerNode(config=PerNode.Config()).clients(source=source)
    assert [(name, rows["date"].to_list()) for _, name, rows in clients] == [
        ("Net/node_0", [1, 2]),
        ("Net/node_1", [1, 2]),
    ]
    assert all("node" not in rows.columns for _, _, rows in clients)
    assert len(PerNode(config=PerNode.Config(nodes=(1,))).clients(source=source)) == 1
    with pytest.raises(ValueError, match="node 3"):
        PerNode(config=PerNode.Config(nodes=(3,))).clients(source=source)
    with pytest.raises(ValueError, match="no node column"):
        PerNode(config=PerNode.Config()).clients(
            source=SimpleNamespace(clients=[("Net", frame.drop("node"))])
        )


def test_jena_weather_renames_unit_headers_and_drops_repeated_dates(
    workspace_tmp_path: Path,
) -> None:
    from kairots.data.sources.JenaWeather import JenaWeather

    channels = JenaWeather.Config().input_channels
    units = {"max_wv": "max. wv (m/s)", "SWDR": "SWDR (W/m\xb2)"}
    headers = [units.get(channel, f"{channel} (unit)") for channel in channels]
    rows = [
        "2020-01-01 00:10:00",
        "2020-01-01 00:20:00",
        "2020-01-01 00:20:00",
        "2020-01-01 00:30:00",
    ]
    lines = [",".join(["date", *headers])]
    lines += [",".join([date] + ["1.0"] * len(headers)) for date in rows]
    raw = workspace_tmp_path / "JenaWeather"
    raw.mkdir()
    # Latin-1 like TSLib's file, so the unit byte is not valid UTF-8.
    (raw / "weather.csv").write_bytes(("\n".join(lines) + "\n").encode("latin-1"))
    source = JenaWeather.load(config=JenaWeather.Config(raw_path=str(raw)))
    ((name, frame),) = source.clients
    assert name == "weather"
    assert frame.columns == ["date", *channels]
    assert frame.height == 3


def test_google_stock_dates_trading_days_and_computes_log_returns(
    workspace_tmp_path: Path,
) -> None:
    from kairots.data.sources.GoogleStock import GoogleStock

    raw = workspace_tmp_path / "GoogleStock"
    raw.mkdir()
    channels = GoogleStock.Config().input_channels
    lines = [",".join(channels)] + [
        ",".join([str(price)] * len(channels)) for price in (100.0, 110.0, 99.0)
    ]
    (raw / "stock_data.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    prices = GoogleStock.load(config=GoogleStock.Config(raw_path=str(raw)))
    ((_, frame),) = prices.clients
    assert frame.get_column("date").to_list() == [0, 1, 2]
    assert frame.get_column("Close").to_list() == [100.0, 110.0, 99.0]
    returns = GoogleStock.load(
        config=GoogleStock.Config(raw_path=str(raw), returns=True)
    )
    ((_, frame),) = returns.clients
    assert frame.get_column("date").to_list() == [0, 1]
    assert frame.get_column("Volume").to_list() == pytest.approx(
        [math.log(1.1), math.log(0.9)]
    )


def _stations(*, root: Path) -> Path:
    """Two WEATHER-5K-shaped stations; station b lacks 02:00 and is stored reversed."""
    root.mkdir()
    for station, base, dates in (
        ("a", 10.0, ["00", "01", "02"]),
        ("b", 20.0, ["01", "00"]),
    ):
        count = len(dates)
        pl.DataFrame(
            {
                "DATE": [f"2014-01-01 {hour}:00:00" for hour in dates],
                "TMP": [base + int(hour) for hour in dates],
                "DEW": [1.0] * count,
                "WND_ANGLE": [90.0] * count,
                "WND_RATE": [3.0] * count,
                "SLP": [1000.0] * count,
                "MASK": ["[1 1 1 1 1]"] * count,
            }
        ).write_csv(root / f"{station}.csv")
    return root


def test_row_concat_splits_each_channel_into_a_univariate_client(
    workspace_tmp_path: Path,
) -> None:
    from kairots.data.sources.RowConcat import RowConcat

    raw = _stations(root=workspace_tmp_path / "TinyWeather5K")
    config = RowConcat.Config(
        member="TinyWeather5K", raw_path=str(raw), channels=("TMP", "SLP")
    )
    source = RowConcat.load(config=config)
    assert [name for name, _ in source.clients] == [
        "TinyWeather5K/a/TMP",
        "TinyWeather5K/a/SLP",
        "TinyWeather5K/b/TMP",
        "TinyWeather5K/b/SLP",
    ]
    assert all(frame.columns == ["date", "value"] for _, frame in source.clients)
    assert source.clients[0][1]["value"].to_list() == [10.0, 11.0, 12.0]
    assert source.clients[2][1]["value"].to_list() == [21.0, 20.0]
    with pytest.raises(ValueError, match="no channels"):
        RowConcat.load(
            config=RowConcat.Config(
                member="TinyWeather5K", raw_path=str(raw), channels=("rain",)
            )
        )


def test_column_concat_joins_clients_on_shared_dates(workspace_tmp_path: Path) -> None:
    from kairots.data.sources.ColumnConcat import ColumnConcat

    raw = _stations(root=workspace_tmp_path / "TinyWeather5K")
    channels = ("TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP")
    names = tuple(f"{station}/{channel}" for station in "ab" for channel in channels)
    source = ColumnConcat.load(
        config=ColumnConcat.Config(raw_path=str(raw), input_channels=names)
    )
    ((name, frame),) = source.clients
    assert name == "TinyWeather5K" and frame.columns == ["date", *names]
    # Only the dates both stations have, in time order, each station's own values.
    assert frame["date"].dt.hour().to_list() == [0, 1]
    assert frame["a/TMP"].to_list() == [10.0, 11.0]
    assert frame["b/TMP"].to_list() == [20.0, 21.0]
    with pytest.raises(ValueError, match="differ from input_channels"):
        ColumnConcat.load(
            config=ColumnConcat.Config(raw_path=str(raw), input_channels=names[:5])
        )


def test_feature_join_matches_keyed_ground_truth(workspace_tmp_path: Path) -> None:
    from kairots.data.sources.FeatureJoin import FeatureJoin

    ett = workspace_tmp_path / "ETDataset"
    (ett / "ETTh1").mkdir(parents=True)
    pl.DataFrame(
        {
            "date": [f"2016-07-01 0{hour}:00:00" for hour in range(4)],
            "OT": [0.0, 1.0, 2.0, 3.0],
        }
    ).write_csv(ett / "ETTh1" / "ETTh1.csv")
    electricity = workspace_tmp_path / "Electricity"
    electricity.mkdir()

    def write(*, hours: list[int]) -> None:
        pl.DataFrame(
            {
                "date": [f"2016-07-01 0{hour}:00:00" for hour in hours],
                "OT": [100.0 + hour for hour in hours],
            }
        ).write_csv(electricity / "electricity.csv")

    # Reordered, 02:00 missing, 05:00 unmatched.
    write(hours=[3, 1, 0, 5])
    config = FeatureJoin.Config(raw_paths=(str(ett), str(electricity)))
    ((name, frame),) = FeatureJoin.load(config=config).clients
    assert name == "FeatureJoin"
    assert frame.columns == ["date", "ETTh1/OT", "Electricity/OT"]
    assert frame["date"].dt.hour().to_list() == [0, 1, 3]
    assert frame["ETTh1/OT"].to_list() == [0.0, 1.0, 3.0]
    assert frame["Electricity/OT"].to_list() == [100.0, 101.0, 103.0]
    write(hours=[0, 1, 1])
    with pytest.raises(ValueError, match="duplicate dates"):
        FeatureJoin.load(config=config)
    with pytest.raises(ValueError, match="no channel of Electricity"):
        FeatureJoin.load(
            config=FeatureJoin.Config(
                raw_paths=config.raw_paths, input_channels=("ETTh1/OT",)
            )
        )


def test_electricity_reads_tslib_columns(workspace_tmp_path: Path) -> None:
    from kairots.data.sources.Electricity import Electricity

    channels = Electricity.Config().input_channels
    assert len(channels) == 321 and channels[-1] == "OT"
    raw = workspace_tmp_path / "Electricity"
    raw.mkdir()
    lines = [",".join(["date", *channels])]
    lines += [
        ",".join([f"2016-07-01 0{hour}:00:00", *["1.0"] * len(channels)])
        for hour in (2, 3)
    ]
    (raw / "electricity.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ((name, frame),) = Electricity.load(
        config=Electricity.Config(raw_path=str(raw))
    ).clients
    assert name == "electricity" and frame.columns == ["date", *channels]
    assert frame.height == 2


def test_tourism_large_reads_the_summing_matrix_and_checks_sums(
    workspace_tmp_path: Path,
) -> None:
    from kairots.data.sources.TourismLarge import TourismLarge

    root = workspace_tmp_path / "hierarchical" / "TourismLarge"
    root.mkdir(parents=True)
    pl.DataFrame(
        {
            "": ["1998-01-01", "1998-02-01"],
            "TotalAll": [3.0, 7.0],
            "AAAHol": [1.0, 3.0],
            "AAAVis": [2.0, 4.0],
        }
    ).write_csv(root / "data.csv")
    pl.DataFrame(
        {
            "": ["TotalAll", "AAAHol", "AAAVis"],
            "AAAHol": [1, 1, 0],
            "AAAVis": [1, 0, 1],
        }
    ).write_csv(root / "agg_mat.csv")
    config = TourismLarge.Config(
        raw_path=str(workspace_tmp_path), input_channels=("TotalAll", "AAAHol")
    )
    source = TourismLarge.load(config=config)
    ((_, frame),) = source.clients
    assert frame.columns == ["date", "TotalAll", "AAAHol"]
    assert frame["date"].dt.month().to_list() == [1, 2]
    assert source.manifest_fields["bottom"] == ["AAAHol"]
    assert source.manifest_fields["summing"] == {
        "TotalAll": ["AAAHol", "AAAVis"],
        "AAAHol": ["AAAHol"],
    }
    pl.DataFrame(
        {"": ["1998-01-01"], "TotalAll": [4.0], "AAAHol": [1.0], "AAAVis": [2.0]}
    ).write_csv(root / "data.csv")
    with pytest.raises(ValueError, match="sums of their bottom members"):
        TourismLarge.load(config=config)


def test_standard_scaler_skips_unobserved_values() -> None:
    from kairots.scalers.Standard import Standard

    nan = float("nan")
    values = torch.tensor([[1.0, nan, nan], [3.0, 5.0, nan], [nan, 5.0, nan]])
    scaler = Standard.fit(values=values, config=Standard.Config(imputation="masked"))
    assert scaler.mean.tolist() == [2.0, 5.0, 0.0]
    assert scaler.scale.tolist() == [1.0, 1.0, 1.0]


def test_physionet_2012_rows_are_observation_times_with_masks(
    workspace_tmp_path: Path,
) -> None:
    import io
    import tarfile

    from kairots.data.sources.PhysioNet2012 import PhysioNet2012

    records = {
        "a": {
            1: "Time,Parameter,Value\n00:00,RecordID,1\n00:00,Age,70\n"
            "00:00,Height,-1\n00:00,ICUType,3\n00:30,HR,80\n00:30,HR,90\n"
            "02:15,Temp,37\n",
            2: "Time,Parameter,Value\n00:00,RecordID,2\n00:00,ICUType,1\n01:00,HR,60\n",
        },
        "b": {3: "Time,Parameter,Value\n00:00,RecordID,3\n00:00,ICUType,3\n"},
    }
    for letter, files in records.items():
        with tarfile.open(workspace_tmp_path / f"set-{letter}.tar.gz", "w:gz") as tar:
            for record, text in files.items():
                member = tarfile.TarInfo(f"set-{letter}/{record}.txt")
                member.size = len(text.encode())
                tar.addfile(member, io.BytesIO(text.encode()))
        (workspace_tmp_path / f"Outcomes-{letter}.txt").write_text(
            "RecordID,In-hospital_death\n"
            + "".join(f"{record},{record % 2}\n" for record in files)
        )
    config = PhysioNet2012.Config(
        raw_path=str(workspace_tmp_path),
        input_channels=("time", "Age", "Height", "HR", "Temp"),
        input_length=3,
    )
    clients = dict(PhysioNet2012.load(config=config).clients)
    assert sorted(clients) == ["CardiacSurgery", "Coronary", "Medical", "Surgical"]
    medical = clients["Medical"]
    assert medical["sample"].to_list() == [0] * 3 + [2] * 3
    assert medical["fold"].to_list() == ["train"] * 3 + ["test"] * 3
    assert medical["label"].to_list() == [1] * 6
    assert medical["time"].to_list() == [0.0, 0.5, 2.25, None, None, None]
    assert medical["Age"].to_list() == [70.0, None, None, None, None, None]
    assert medical["Height"].null_count() == 6
    assert medical["HR"].to_list() == [None, 85.0, None, None, None, None]
    # RecordID and ICUType are not variables, so the first row is 01:00.
    assert clients["Coronary"]["time"].to_list() == [1.0, None, None]
    assert clients["Coronary"]["HR"].to_list() == [60.0, None, None]
    assert clients["Surgical"].height == 0
    with pytest.raises(ValueError, match="observation times"):
        PhysioNet2012.load(
            config=PhysioNet2012.Config(**{**vars(config), "input_length": 2})
        )


def test_csdi_physionet_hourly_source_and_patient_folds(
    workspace_tmp_path: Path,
) -> None:
    import io
    import tarfile

    from kairots.data.sources.PhysioNet2012.PhysioNet2012CSDI import (
        PhysioNet2012CSDI,
    )
    from kairots.data.splits.CSDIFiveFold import CSDIFiveFold

    archive = workspace_tmp_path / "set-a.tar.gz"
    text = (
        "Time,Parameter,Value\n00:10,HR,80\n00:50,HR,90\n01:05,HR,100\n01:40,Temp,37\n"
    )
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("set-a/1.txt")
        member.size = len(text.encode())
        tar.addfile(member, io.BytesIO(text.encode()))
        empty = "Time,Parameter,Value\n00:00,RecordID,2\n"
        member = tarfile.TarInfo("set-a/2.txt")
        member.size = len(empty.encode())
        tar.addfile(member, io.BytesIO(empty.encode()))
    loaded = PhysioNet2012CSDI.load(
        config=PhysioNet2012CSDI.Config(
            raw_path=str(workspace_tmp_path), normalization="train_only"
        )
    )
    frame = loaded.clients[0][1]
    assert frame.height == 96
    assert frame.get_column("HR").head(2).to_list() == [90.0, 100.0]
    assert frame.get_column("Temp").head(2).to_list() == [None, 37.0]
    assert len(loaded.input_channels) == 35
    assert frame.filter(pl.col("sample") == 1).select("HR").null_count().item() == 48
    historical = PhysioNet2012CSDI.load(
        config=PhysioNet2012CSDI.Config(raw_path=str(workspace_tmp_path))
    ).clients[0][1]
    assert historical.get_column("HR").head(2).to_list() == [-1.0, 1.0]

    samples = pl.DataFrame(
        {"sample": range(10), "fold": ["train"] * 10, "label": [0] * 10}
    )
    policy = CSDIFiveFold(config=CSDIFiveFold.Config(fold=2))
    splits = policy.split(samples=samples)
    assert len(splits["test"]) == 2
    assert len(splits["validation"]) == 1
    assert len(splits["train"]) == 7
    assert not set(splits["train"]) & set(splits["validation"])
    assert not set(splits["train"]) & set(splits["test"])
    indices = np.arange(10)
    np.random.RandomState(1).shuffle(indices)
    test = indices[4:6]
    remaining = np.delete(indices, np.arange(4, 6))
    np.random.RandomState(1).shuffle(remaining)
    assert splits["test"] == test.tolist()
    assert splits["train"] == remaining[:7].tolist()
    assert splits["validation"] == remaining[7:].tolist()


def test_complete_window_ratio_counts_month_starts_in_months() -> None:
    from datetime import datetime

    months = [datetime(1998 + index // 12, index % 12 + 1, 1) for index in range(24)]
    frame = pl.DataFrame({"date": months[:10] + months[11:], "v": np.ones(23)})
    splits = CompleteWindowRatio().split(
        frame=frame,
        input_length=3,
        output_length=2,
        offset_length=0,
        ratios=(0.6, 0.0, 0.4),
        channels=("v",),
        lookback_crosses_boundary=True,
        min_windows=1,
    )
    starts = [start for name in ("train", "test") for start, _ in splits[name]]
    # 23 months with November 1998 missing: windows of 5 months avoid the gap.
    assert starts and all(start + 5 <= 10 or start >= 10 for start in starts)
    assert len(starts) <= 6 + 9
    assert any(start >= 10 for start in starts)


def test_store_code_digest_follows_the_defining_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import sys

    from kairots.data.stores.Parquet import code

    def load(*, text: str) -> type:
        path = tmp_path / "stub_source.py"
        path.write_text(text, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("stub_source", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "stub_source", module)
        spec.loader.exec_module(module)
        return module.Stub

    first = code(types=(load(text="class Stub:\n    VALUE = 1\n"),))
    assert code(types=(load(text="class Stub:\n    VALUE = 1\n"),)) == first
    assert code(types=(load(text="class Stub:\n    VALUE = 2\n"),)) != first
    # A composite source's digest follows its members' files too.
    composite = type(
        "Composite", (), {"MEMBERS": (load(text="class Stub:\n    VALUE = 1\n"),)}
    )
    joined = code(types=(composite,))
    setattr(composite, "MEMBERS", (load(text="class Stub:\n    VALUE = 3\n"),))
    assert code(types=(composite,)) != joined


def test_exchange_rate_reads_eight_tslib_columns(workspace_tmp_path: Path) -> None:
    from kairots.data.sources.ExchangeRate import ExchangeRate

    path = workspace_tmp_path / "exchange_rate.csv"
    path.write_text("date,0,1,2,3,4,5,6,OT\n1990/1/1 0:00,1,2,3,4,5,6,7,8\n")
    source = ExchangeRate.load(
        config=ExchangeRate.Config(raw_path=str(workspace_tmp_path))
    )
    assert source.input_channels == ("0", "1", "2", "3", "4", "5", "6", "OT")
    assert source.clients[0][1].height == 1
    assert source.clients[0][1].schema["date"] == pl.Datetime("us")


@pytest.mark.parametrize(("input_length", "output_length"), [(336, 96), (96, 96)])
def test_cut_rows_reproduce_the_ett_loader_windows(
    *, input_length: int, output_length: int
) -> None:
    month = 30 * 24
    borders = {
        "train": (0, 12 * month),
        "validation": (12 * month - input_length, 16 * month),
        "test": (16 * month - input_length, 20 * month),
    }
    # The ETT loaders enumerate every window inside each border pair.
    expected = {
        name: [
            (start, start + input_length)
            for start in range(first, last - input_length - output_length + 1)
        ]
        for name, (first, last) in borders.items()
    }
    # The raw file runs past the last border (ETTh1 has 17420 rows).
    frame = _frame(count=17420).with_columns(pl.col("date").cast(pl.Int64))
    policy = CompleteWindowRatio(
        config=CompleteWindowRatio.Config(cut_rows=(12 * month, 16 * month, 20 * month))
    )
    splits = policy.split(
        frame=frame,
        input_length=input_length,
        output_length=output_length,
        offset_length=0,
        ratios=(0.6, 0.2, 0.2),
        channels=("v",),
        lookback_crosses_boundary=True,
        min_windows=1,
    )
    assert splits == expected


def test_drop_last_drops_only_the_short_training_batch() -> None:
    from kairots.data.views.Offline import Offline

    values = torch.arange(60.0).reshape(30, 2)
    options = {
        "values": values,
        "windows": [(start, start + 2) for start in range(10)],
        "input_length": 2,
        "output_length": 1,
        "offset_length": 0,
        "batch_size": 4,
        "generator": torch.Generator().manual_seed(0),
    }
    view = Offline(config=Offline.Config(drop_last=True))
    assert [len(x) for x, _ in view.batches(**options, shuffle=True)] == [4, 4]
    assert [len(x) for x, _ in view.batches(**options, shuffle=False)] == [4, 4, 2]
