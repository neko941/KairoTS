# Getting started

## Install

KairoTS needs Python 3.14. From the repository root:

```bash
pip install -e ".[dev,docs]"   # dev: pytest, ruff, pyright; docs: Sphinx
pip install -e ".[estimators]"  # optional: xgboost for the XGBoost model
```

Install the PyTorch build that matches your hardware first if the default wheel does not.

## Data

Sources with published download URLs fetch their raw files into
`datasets/<Family>/raw/`, check their sha256, and never modify them. TinyWeather5K
requires station CSVs to be placed there by hand; see the
[dataset guide](customization/datasets.md). Prepared data goes to
`datasets/<Family>/prepared/`. To prepare it without training:

```bash
python -m kairots prepare --config configs/forecasting/ETDatasetHour.json
```

## Run an experiment

One config per task and dataset lives under `configs/<task>/`. A command-line option
`--namespace.field value` overrides any config field:

```bash
# Forecasting: ETTh1 and ETTh2 as two clients
python -m kairots --config configs/forecasting/ETDatasetHour.json --model.name DLinear

# Classification: BasicMotions dealt IID to two clients
python -m kairots --config configs/classification/BasicMotions.json --strategy.iterations 5
```

Each run writes a self-contained directory under `runs/<experiment>/<time>-<hash>/`:
`config.json`, `manifest.json` (raw-file hashes, clients, splits, exclusions),
`rounds.jsonl`, `clients.jsonl`, `summary.json` and optional `checkpoints/`.

## Separate source fitting and target evaluation

The Python API can fit a source model without exposing its test windows to the
workers, then evaluate a completed fit artifact on a target query split:

```python
import json
from pathlib import Path

from kairots import Experiment, evaluate, fit

values = json.loads(Path("configs/forecasting/ETDatasetHour.json").read_text())
source = Experiment.from_dict(values=values)
fitted = fit(experiment=source)
target = Experiment.from_dict(values=values)
result = evaluate(experiment=target, source_run=fitted.run_dir)
```

Fitting uses `train` as source data and `validation` as support for checkpoint
selection. Evaluation loads the selected source checkpoint and queries only `test`;
target training is disabled. For TTA, protocols see query inputs without query
labels, while prequential evaluation releases a target only after its prediction.
The run manifest records the phase permissions. The full CLI `run()` path remains
available for combined training and evaluation.

## Runnable examples and expected outputs

These configs demonstrate one supported flow for each task type. They are
engineering examples; the paper-result comparisons and their protocol limits are
listed on the corresponding method pages.

| Task | Config | Expected run output |
|---|---|---|
| Forecasting | `configs/forecasting/ETDatasetHour.json` | Per-round and final configured forecast metrics; checkpoint files when enabled |
| Classification | `configs/classification/BasicMotions.json` | Per-round and final classification metrics and selected checkpoint |
| Imputation | `configs/imputation/ETDatasetHour.json` | Per-round and final reconstruction metrics, evaluated on the configured missing entries |
| Anomaly detection | `configs/anomaly_detection/SMDAll-TimesNet-Historical.json` | F1 and AdjustedF1 under the explicitly labeled historical threshold protocol |
| Likelihood forecasting | `configs/likelihood_forecasting/ETDatasetHour.json` | Gaussian objective and configured forecast metrics |
| Probabilistic forecasting | `configs/probabilistic_forecasting/ETDatasetHour.json` | Configured distribution/quantile scores |
| Sample forecasting | `configs/sample_forecasting/ExchangeRate-TimeGrad.json` | Sample forecast scores; the example is not a TimeGrad paper reproduction |
| Probabilistic imputation | `configs/probabilistic_imputation/ETDatasetHour-CSDI.json` | Configured probabilistic imputation scores; the example is not a CSDI paper reproduction |
| Generation | `configs/generation/ETDatasetHour.json` | Configured sample-quality scores |
| Control | `configs/control/ETDatasetHour-Battery.json` | Episode return and configured control scores |

Every completed run has these artifacts:

- `config.json`: resolved component settings, seed, and software versions.
- `manifest.json`: raw-file hashes, channel order, clients, split/scaler data,
  exclusions, and initialization details.
- `rounds.jsonl` and `clients.jsonl`: pooled/per-client metrics, timing, and
  downlink, uplink and peer bytes (per round and per client).
- `summary.json`: completion status, final metrics, checkpoint-selection record,
  total time, bytes, and `model`: parameter, memory and mult-add counts of the
  global model on one training batch (`mult_adds_torchinfo` follows torchinfo,
  `mult_adds` is the FLOP count halved).
- `checkpoints/final.pt` when checkpoint saving is enabled; with validation-based
  selection, `checkpoints/best_val.pt` is also written. Continual runs may add
  `continual.json`.

Run one config with `python -m kairots --config <path>` and inspect its saved
directory with `python -m kairots report <runs directory>`.

## Quick test

Run every config for one iteration on the CPU:

```bash
python scripts/smoke.py                  # all tasks
python scripts/smoke.py --task classification
```

## See what you can choose

```bash
python -m kairots list                   # every category and component
python -m kairots list metrics --task Classification
```

The [component catalog](reference/catalog.md) has the same list. To add your own
component, see [Customization](customization/index.md).
