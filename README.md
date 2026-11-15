# KairoTS

A modular framework for federated time-series learning: pick a task, dataset,
partition, split, model and strategy by name, and run it on one process or across Ray
workers.

**Status:** early development (0.0.0). The APIs change without notice.

| Task | Datasets | Models | Strategies |
|---|---|---|---|
| Forecasting | ETTh1, ETTh2, ETDatasetHour; ExchangeRate; M4Hourly (SMAPE, MASE); PeMS08 (one client per sensor); TourismLarge (hierarchy, summing matrix) | Linear, DLinear, TimesNet, Naive, TiDE (calendar and past covariates), SplitNN (vertical parties) | FedAvg, LocalOnly, Centralized, Ditto, SplitNN (vertical), Transfer (pretrained source, few-shot target) |
| Classification | BasicMotions (UEA); PhysioNet2012 (irregular, masks) | DLinear, TimesNet, GRUD (irregular) | FedAvg, LocalOnly, Centralized, Ditto, FLwF2 (class-incremental) |
| Imputation | ETTh1, ETTh2, ETDatasetHour | Linear, DLinear, TimesNet | FedAvg, LocalOnly, Centralized |
| Probabilistic forecasting (quantiles) | ETTh1, ETTh2, ETDatasetHour | DLinear | FedAvg, LocalOnly, Centralized |
| Gaussian likelihood forecasting | ETTh1, ETTh2, ETDatasetHour | DLinear | FedAvg, LocalOnly, Centralized |
| Diffusion forecasting (joint samples) | ExchangeRate | TimeGrad | Centralized, FedAvg |
| Probabilistic imputation | ETDatasetHour | CSDI | Centralized, FedAvg |
| Anomaly detection | SMDGroup1 (SMD) | Linear, DLinear, TimesNet | FedAvg, LocalOnly, Centralized |
| Control (Battery environment, REINFORCE learner) | ETTh1 (`HUFL` as the price) | Policy | Centralized |
| Generation (synthesis, DDPM) | ETTh1, ETTh2, ETDatasetHour; GoogleStock (log returns, stylized facts) | Denoiser | FedAvg, LocalOnly, Centralized |

`python -m kairots list` prints everything you can select; `python -m kairots report [ROOT]` summarizes completed runs into a table.

## Install

```bash
pip install -e ".[dev,docs]"
pip install -e ".[estimators]"   # optional: xgboost for the XGBoost model
```

KairoTS needs Python 3.14. Raw data downloads on first use into `datasets/`.

## Quickstart

```bash
python -m kairots --config configs/forecasting/ETDatasetHour.json
python -m kairots --config configs/classification/BasicMotions.json --strategy.iterations 5
python -m kairots --config configs/forecasting/M4Hourly.json
python -m kairots --config configs/forecasting/PeMS08.json
python -m kairots --config configs/imputation/ETDatasetHour.json
python -m kairots --config configs/anomaly_detection/SMDGroup1.json
python scripts/smoke.py          # every config, one iteration, CPU
python scripts/leakage.py        # training is identical with a poisoned test set
```

Any config field can be overridden as `--namespace.field value`. A component's
singleton requirements fill omitted fields; explicit conflicting values are rejected.
Requirements with several allowed values never choose one. Each run writes a
self-contained directory under `runs/`. An interrupted run continues from its last
finished round with the same config plus `--runtime.resume <run directory>`.

Diffusion tasks and models use `diffusion_steps` for the noise-grid length.
XGBoost's boosting coefficient is `model.shrinkage`; gradient optimization uses
`learner.learning_rate`.

`configs/forecasting/ExchangeRate-ZeroShot.json` reuses a compatible completed
source run or trains its `strategy.source_config` first, then scores the target
without updates. Set `initialization.run_dir` to select a saved source explicitly.

Ordinary training uses the full training split through `Offline`. Select
`training: {"name": "ClassIncremental", "periods": 2}` for class-incremental
classification. `Continual` is the non-selectable base for period schedules.

`Informer` reads history and future calendar marks, with zero future values in its
decoder. Its calendar order and source choices are documented under
[Models](docs/customization/models.md) and [design notes](docs/explanation/notes.md).

## Layout

```text
configs/<task>/   runnable configs     scripts/   format, docs, smoke, clean
src/kairots/      the package          tests/     unit, integration, ray
docs/             Sphinx site
```

## Documentation

Build it with `python scripts/docs.py` and open `docs/_build/html/index.html`.

- [Getting started](docs/getting-started.md)
- [Customization](docs/customization/index.md): add a model, metric, dataset, ...
- [Architecture](docs/explanation/architecture.md) and [design notes](docs/explanation/notes.md)
- [Contributing](docs/contributing.md)

## License

[MIT-0](LICENSE): anyone may use, copy, modify and distribute it, with no conditions.

Objectives and reported scores share `src/kairots/metrics/`; `ROLES` declares
whether a component supports a batch loss, a mergeable metric, or both.

Loss-landscape analysis runs after training: install `pip install -e ".[analysis]"`,
then run `python -m kairots landscape RUN_DIR --dimensions 2`. It saves a contour
and a 3D surface from the same grid. `--dimensions 1` produces a loss curve.
No extra training logs are required; keep the checkpoint and prepared data.
