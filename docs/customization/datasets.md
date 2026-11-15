# Datasets

A dataset is a **source**. Sources form a hierarchy by structure kind, then family, then
variant; only concrete variants are selectable.

**Copy:** `src/kairots/data/sources/ETDataset/ETTh1.py` (family: `ETDataset/__init__.py`)

```text
Source (base.py)
  Multivariate          aligned channels on one regular time grid (series)
    ETDataset/          family: shared schema and CSV reader
      ETTh1, ETTh2      variants: one file each
      ETDatasetHour     composite of ETTh1 + ETTh2 (one client each)
    JenaWeather         single file: one station, 21 channels, 10-minute steps
    BeijingAirQuality   UCI 501: 12 stations as clients, 11 hourly channels, NaN where missing
    TinyWeather5K       WEATHER-5K station files placed by hand, one client each
    Electricity         TSLib's ECL: 321 households as channels, hourly
    RowConcat           each channel of a member source as its own univariate client
    ColumnConcat        a member source's clients as channels of one client, joined on date
    FeatureJoin         one-client sources joined on exact dates as one client's channels
    Hierarchical        every series of a hierarchy as channels, with its summing matrix
      TourismLarge      555 monthly Australian tourism series, 304 at the bottom level
  Collection            differently sized univariate series, one client each
    M3/                   family: original competition `.xls` actuals and held-out horizons
      M3Yearly            645 yearly series, 6-step test horizon
      M3Quarterly         756 quarterly series, 8-step test horizon
      M3Monthly           1,428 monthly series, 18-step test horizon
      M3Other             174 other-frequency series, 8-step test horizon
    M4/                  family: official train/test folds and M4 metadata
      M4Yearly           23,000 series, 6-step test horizon
      M4Quarterly        24,000 series, 8-step test horizon
      M4Monthly          48,000 series, 18-step test horizon
      M4Weekly           359 series, 13-step test horizon
      M4Daily            4,227 series, 14-step test horizon
      M4Hourly           414 series, 48-step test horizon
    M5/                  family: official grouped retail-sales folds and weights
      M5Validation       30,490 product-store series, 28-day public holdout
      M5Evaluation       30,490 product-store series, 28-day final holdout
  Panel                 many equal-length labeled sequences (classification)
    UEA/                family: `.ts` reader, mirrors, official TRAIN/TEST folds
      BasicMotions      variant: 6 channels, length 100, 4 classes
      EthanolConcentration variant: 3 channels, length 1751, 4 classes
    Irregular           rows are a sample's own observation times, NaN where unobserved
      PhysioNet2012     ICU mortality: 8000 patients, 40 variables, one client per ICU type
  SpatioTemporal        sensor networks recorded over time
    PeMS/               family: traffic sensor arrays and road graphs
      PeMS08            170 sensors, three channels, one network frame
```

A series source returns frames with `date` and the channels. A panel source returns
one row per sample and step: `sample, step, fold, label` and the channels.

## Contract

| Member | Purpose |
|---|---|
| `Config` | subclass the family's `Config`; `raw_path`, `input_channels`, `batch_size`; panel sources add `input_length`, `num_classes` |
| `remote_files(*, config) -> tuple[RemoteFile, ...]` | where to download the raw files (mirrors, sha256) |
| `raw_files(*, config) -> list[Path]` | files read, known without reading them (cache key) |
| `load(*, config) -> Source` | read raw files; return clients as `(name, frame)` |

A new variant of an existing family is usually two constants (`FILE_NAME`, `FREQUENCY`)
and a `Config` with its `name`. Raw files are read-only; the Parquet store prepares them
once under `datasets/` and rebuilds only when a raw file or the config changes.

## Select it

```json
"data": {"source": {"name": "ETTh1", "raw_path": "datasets/ETDataset/raw"}}
```

Run `python -m kairots prepare --config <file>` to build the store without training.

M5 stores the 30,490 bottom-level product-store series as one multivariate client.
Its generated `series_00000`-style channel names retain their original product and
store identifiers in the source manifest. `M5WRMSSE` aggregates bottom-level
predictions through the official 12-level hierarchy and uses the corresponding
released fold weights. The validation and final evaluation variants use separate
official 28-day actuals and weight files. The `M5Evaluation-Naive` config is a
reproducible check of the final-fold data, one-step persistence baseline, and
WRMSSE metric against the paper's reported Naive result.
