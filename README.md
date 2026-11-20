# KairoTS

[![Latest tag](https://img.shields.io/github/v/tag/neko941/KairoTS?sort=semver&label=version)](https://github.com/neko941/KairoTS/tags)
[![Tests](https://img.shields.io/github/actions/workflow/status/neko941/KairoTS/tests.yml?branch=main&label=tests)](https://github.com/neko941/KairoTS/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.14-blue)](pyproject.toml)
[![License: MIT-0](https://img.shields.io/badge/license-MIT--0-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/neko941/KairoTS?style=flat)](https://github.com/neko941/KairoTS/stargazers)

A modular framework for federated time-series learning.
Pick a task, dataset, partition, split, model and strategy by name, and run it on one process or across Ray workers.
Every method follows its paper exactly; where the paper is silent, the official code decides.

**Status:** early development.
The APIs change without notice.

## Install

```bash
pip install -e ".[dev,docs]"
pip install -e ".[estimators]"   # optional: xgboost for the XGBoost model
pip install -e ".[analysis]"     # optional: matplotlib for loss landscapes
```

KairoTS needs Python 3.14.
Raw data downloads on first use into `datasets/`.

## Quickstart

```bash
# Reproduce one published table cell: DLinear, ETTh1, horizon 96
python -m kairots --config configs/zeng2023-dlinear/Table2-ETTh1.json --run DLinear-96

# Federated demos
python -m kairots --config configs/examples/forecasting/ETDatasetHour.json
python -m kairots --config configs/examples/classification/BasicMotions.json --strategy.iterations 5

python -m kairots list           # every selectable component
python -m kairots report runs    # summary table of completed runs
```

Any config field can be overridden as `--namespace.field value`.
Each run writes a self-contained directory under `runs/`; an interrupted run continues with the same config plus `--runtime.resume <run directory>`.

## Configs

Configs are organized by paper, one file per table and dataset (and model, when a table's models differ in settings).
A file holds the shared settings once and sweeps only what changes (horizon, fold, ...) and seeds.

```text
configs/
  <author><year>-<method>/Table<N>[-<Model>][-<Dataset>].json   one table on one dataset
  examples/<task>/<name>.json                     feature demos and settings no paper reports
```

```json
{
  "reproduces": {"paper": "arXiv:2205.13504v3", "code": "https://github.com/cure-lab/LTSF-Linear/tree/0c113668"},
  "seeds": [941, 942, 943],
  "task": {"name": "Forecasting"}, "data": {"...": "..."}, "model": {"name": "DLinear"},
  "sweep": {
    "data.split.output_length": [96, 192, 336, 720],
    "data.source.batch_size": [8, 8, 32, 32]
  }
}
```

`sweep` maps dotted fields (the same as command-line overrides) to equal-length lists, zipped: run i takes every field's i-th value and is named by the first field's value.
A list of such maps runs their product, named by the parts joined with `-` (`"sweep": [{"model.name": ["Linear", "DLinear"]}, {...}]` gives `DLinear-96`).
`--config FILE` runs every sweep entry and seed; `--run 96` and `--seed 941` select one of each.

| Paper | Config · run | Cell | Paper value |
|---|---|---|---|
| DLinear (Zeng et al., AAAI 2023) | `zeng2023-dlinear/Table2-ETTh1` · DLinear-96 | ETTh1, horizon 96, MSE / MAE | 0.375 / 0.399 |
| NLinear (Zeng et al., AAAI 2023) | `zeng2023-dlinear/Table2-ETTh1` · NLinear-96 | ETTh1, horizon 96, MSE / MAE | 0.374 / 0.394 |
| Linear (Zeng et al., AAAI 2023) | `zeng2023-dlinear/Table2-ETTh1` · Linear-96 | ETTh1, horizon 96, MSE / MAE | 0.375 / 0.397 |
| TimesNet (Wu et al., ICLR 2023) | `wu2023-timesnet/Table16-TimesNet` · ETTh1-0.125 | ETTh1 imputation, 12.5% masked, MSE / MAE | 0.057 / 0.159 |
| TimesNet | `wu2023-timesnet/Table5-TimesNet` · SMD | SMD anomaly detection, F1 (%) | 85.81 |
| TimesNet | `wu2023-timesnet/Table17-TimesNet` · EthanolConcentration | EthanolConcentration accuracy (%) | 35.7 |
| TAFAS (Kim et al., AAAI 2025) | `kim2025-tafas/Table1-DLinear` · ETTh1-96 | DLinear on ETTh1-96, frozen / adapted MSE | 0.451 / 0.442 |
| TimeGrad (Rasul et al., ICML 2021) | `rasul2021-timegrad/Table2-TimeGrad` · Electricity | Electricity CRPS-sum | 0.0206 |
| CSDI (Tashiro et al., NeurIPS 2021) | `tashiro2021-csdi/Table1-CSDI` · PhysioNet-0.1 | PhysioNet 10% missing, CRPS | 0.238 |
| M3 competition (Makridakis & Hibon, 2000) | `makridakis2000-m3/Table13-Naive2` · Yearly | Yearly Naive2 sMAPE, horizons 1-6 | 17.88 |
| M5 competition (Makridakis et al., 2022) | `makridakis2022-m5/TableB-Naive` · M5 | Naive WRMSSE | 1.752 |

The documentation's "Paper settings covered" page is generated from these blocks.
Known differences from each paper are recorded in the [design notes](docs/explanation/notes.md).

## Coverage

| Task | Models | Strategies |
|---|---|---|
| Forecasting | Linear, NLinear, DLinear, TimesNet, Naive, TiDE, Informer, iTransformer, iFlashformer, iFlowformer, iInformer, iReformer, SplitNN | FedAvg, LocalOnly, Centralized, Ditto, SCAFFOLD, APFL, SplitNN, Transfer |
| Classification | DLinear, TimesNet, GRUD | FedAvg, LocalOnly, Centralized, Ditto, FedBABU, FML, FLwF2 |
| Imputation | Linear, DLinear, TimesNet | FedAvg, LocalOnly, Centralized |
| Probabilistic and likelihood forecasting | DLinear (quantile and Gaussian heads) | FedAvg, LocalOnly, Centralized |
| Diffusion forecasting | TimeGrad | Centralized, FedAvg |
| Probabilistic imputation | CSDI | Centralized, FedAvg |
| Anomaly detection | Linear, DLinear, TimesNet | FedAvg, LocalOnly, Centralized |
| Generation | Denoiser (DDPM) | FedAvg, LocalOnly, Centralized |
| Control | Policy (REINFORCE, Battery environment) | Centralized |

`python -m kairots list` is the authoritative list, including FedProx, DFedAvg, FedPAQ, ZeroShot and XGBoost.

- `configs/examples/forecasting/ExchangeRate-ZeroShot.json` trains or reuses its `strategy.source_config`, then scores the target without updates.
- `training: {"name": "ClassIncremental", "periods": 2}` selects class-incremental classification.
- `python -m kairots landscape RUN_DIR --dimensions 2` plots the loss surface of a saved run (needs the `analysis` extra).

## Checks

```bash
python scripts/format.py         # format, lint and style
python scripts/smoke.py          # every config, one iteration, CPU
python scripts/leakage.py        # training is identical with a poisoned test set
python -m pytest
```

## Documentation

Build it with `python scripts/docs.py` and open `docs/_build/html/index.html`.

- [Getting started](docs/getting-started.md)
- [Customization](docs/customization/index.md): add a model, metric, dataset, ...
- [Architecture](docs/explanation/architecture.md) and [design notes](docs/explanation/notes.md)
- [Contributing](docs/contributing.md)

## Star history

[![Star history](https://api.star-history.com/svg?repos=neko941/KairoTS&type=Date)](https://star-history.com/#neko941/KairoTS&Date)

## License

[MIT-0](LICENSE): anyone may use, copy, modify and distribute it, with no conditions.
Third-party code keeps its own license; see [NOTICE](NOTICE).
