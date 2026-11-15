# Architecture

## Repository

```text
configs/<task>/<Source>.json   one runnable config per task and dataset
docs/                          this site (Sphinx + MyST)
scripts/                       format, docs, smoke, clean, check_style
src/kairots/                   the package
tests/unit|integration|ray/    fast checks, full runs, Ray parity
datasets/  runs/  tmp/         data, run outputs, scratch (all untracked)
```

## Package

| Path | Role |
|---|---|
| `registry.py` | finds selectable classes by file name, reads their constants with `ast`, imports only the chosen one |
| `config/schema.py` | `Experiment`: one namespace per owner, plus the optional `reproduces` block (paper, table, cells and their printed values, code commit) that marks a config as a paper reproduction, each resolved to its component's nested `Config`; fills model shape fields from the data and the task's default loss and metrics; `groups` (per-group `data` and `client_model`, resolved by `member`) and `client_model` |
| `tasks/` | task semantics: `Forecasting`, `ProbabilisticForecasting` (quantiles), `LikelihoodForecasting` (Gaussian density), `Classification`, `Imputation`, `AnomalyDetection`, `Control` (one action per horizon step, the horizon's prices are the target), `Generation` (DDPM synthesis: the noised window plus a step channel in, the noise out; `sample` runs the reverse process); each declares `HEAD`, `RECONSTRUCTS`, `LOSS`, `METRICS` (and `PROTOCOL` when not `Frozen`) |
| `data/sources/` | one directory per dataset or dataset family, one file per variant (`ETDataset/ETTh1.py`, `SMD/SMDGroup1.py`, `UEA/BasicMotions.py`, `M3/M3Yearly.py`, `M4/M4Yearly.py`, `PeMS/PeMS08.py`); the family class subclasses its structure kind (`Multivariate.py`, `Panel.py`, `Collection.py`: one univariate series per client, `SpatioTemporal.py`: one long `date, node` frame per sensor network with its road graph in the manifest, `Hierarchical.py`: every series of a hierarchy as channels with the summing matrix in the manifest, `Irregular.py`: a panel whose rows are each sample's observation times, NaN where unobserved; `ExchangeRate/__init__.py`: eight TSLib daily currency rates; `GoogleStock/__init__.py`: TimeGAN's daily stock prices, optionally as log returns); `Composite` joins sources |
| `data/download.py` | the only download code: mirrors, retries, resume, sha256, safe extraction |
| `data/partitions/` | rows → clients: `PerFile` (one series file per client), `IID` (samples dealt at random), `Dirichlet` (label-skewed samples), `PerNode` (one client per sensor of a spatio-temporal source) |
| `data/splits/` | train/validation/test: `CompleteWindowRatio` (series windows), `TaskWindowRatio` (the series ratio split inside each time-segment task), `Predefined` (official folds: panel samples or series windows), `Grouped` (ratio split plus a seeded share of held-out clients that never train and are scored as `unseen_*`), `TestFold` (forecast the official test fold from the history before it, M4) |
| `data/stores/Parquet.py` | prepares clients once under `datasets/<Family>/prepared/<data_key>/`; workers get a `ClientRecord` and load lazily |
| `data/views/` | batches from windows: `(start, target)` for series, `(sample, label)` for panel data; `Offline` (shuffled), `Ordered` (time order, for streams) |
| `augmentations/` | optional `data.augmentation` on training inputs only: `Jitter` |
| `scalers/` | `Standard`, fitted on each client's train portion; `Identity` (no scaling) |
| `models/<Model>/` | a recipe with its body and task heads (a head owns its task's parts) (`DLinear`: forecasting, classification, quantile and Gaussian likelihood heads; `TimesNet`: one body for forecasting, reconstruction and classification; `Naive`: last value, no parameters; `TiDE`: reads calendar and past-observed covariates; `XGBoost`: booster bytes in a uint8 buffer; `SplitNN`: one bottom model per channel-owning party and a top model at the label party; `GRUD`: a GRU with input and hidden-state decay that reads the masks and time channel of an irregular panel; `Policy`: an MLP from a window's history to action logits for every horizon step; `Denoiser`: an MLP noise predictor over the flattened window with a sinusoidal step embedding); a model may declare `TASKS`, `STRATEGIES` (strategies that never average it) and `COVARIATES` (it reads the covariate layout of `tasks/Forecasting.py`) |
| `layers/` | building blocks shared by two or more models (`PositionalEmbedding`); a block stays in its model's directory until a second model needs it, and a model never imports from another model's directory |
| `learners/` | local fitting: `Gradient`; `Stateless` (no optimizer, 0 epochs) for parameter-free models such as `Naive`; `Estimator` (one native `model.fit` per epoch, XGBoost); `REINFORCE` (policy gradient on `environments/Battery.py` rewards) |
| `environments/` | decision problems with `reset`/`step` on batched tensors: `Battery` (arbitrage against a load-driven price, charge limits, no randomness) |
| `initializers/` | `base.py` holds the `INITIALIZERS` name table (`torch.nn.init`); policies (`Random` default: constructor values, seeded, model-independent `Average`, `Rules`, `Pretrained` from a saved run's checkpoint) applied once to the canonical global model |
| `schedulers/` | top-level `scheduler`: `Constant` (default), `Cosine`; stepped once per client update, state in `ClientState.scheduler_state` |
| `optimizers/` | nested optimizer configs build `Adam` or `SGD`; weight decay and SGD momentum are configurable; learning rates remain on the learner or adaptation protocol |
| `metrics/` | objectives and mergeable scores declare `ROLES` (`loss`, `metric`, or both), `TASKS` and `NEEDS`; a metric that NEEDS `seasonal_scale` (MASE) also gets each window's mean seasonal difference before its target, from one cumulative sum per client |
| `strategies/` | families `Traditional` (`FedAvg`, `FLwF2` (two frozen classification teachers, previous client model in client state), `ZeroShot` (complete DLinear checkpoint; optional source config reuses or trains a source run before scoring, no target jobs or messages), `SCAFFOLD`: control variates in the downlink, uplink and `ClientState.extra`, `FedBABU`: the head stays at its start and only the body travels, scored after fine-tuning, `Centralized`, `SplitNN`: vertical split learning on one aligned timeline, cut-layer traffic billed), `NonFederated` (`LocalOnly`; `Transfer`: local fine-tuning of a pretrained model with `frozen` parameter patterns) `Personalized` (`Ditto`; `APFL`: each client scores `alpha * v + (1 - alpha) * w`), `Decentralized` (`DFedAvg`, gossip in `exchange`), `Heterogeneous` (client models; `FML`: classification, a private client model and a shared meme model, mutual distillation; its `PartialTraining` branch: `FedPMT`, frozen groups, partial uplink, per-parameter mean), `Quantized` (`FedPAQ`: lossy update codec, billed by `Message.encoded_bytes`) and the local-objective `Proximal` (`FedProx`; `DFedProx` combines it with `Decentralized`, clashes fail at import); a strategy may declare `TASKS`, `MODELS` and `SOURCES` (what it runs on); private per-client parameters travel in `ClientState.model_state` |
| `topologies/` | peer maps for decentralized strategies: `FullyConnected`, `Ring` |
| `evaluation/` | `populations.select` supplies seen/cross-dataset and unseen-client IDs and phases to every scoring protocol; `Frozen` scores the global model per phase, per client and pooled (hooks `start_client`, `predict`), and with `finetune_epochs` also after each client fine-tunes it (`finetuned_*`); `Calibrated` adds anomaly thresholds; `Adapted` runs test-time adaptation on the test phase and reports the frozen scores as `frozen_*`; `Prequential` scores the test stream test-then-train (with `frozen_*`); `TAFAS` adapts gated calibration modules around the frozen model on the test stream's partially observed targets (with `frozen_*`); `BottomUp` replaces every series' forecast by the sum of its bottom members' (hierarchical sources); `Generative` samples as many windows as each client's strided real windows and scores the two sets (MMD, Wasserstein, ACFError, CorrelationError, DiscriminativeScore, and TSTR and DCR against the train windows; a config adds the stylized facts of returns KurtosisError, VolatilityClusteringError, LeverageError, VaRError and ESError) |
| `training/` | `Offline` (implicit or explicit full-training-split selection); `ClassIncremental` (disjoint classification releases and checkpointed replay, inheriting the non-selectable `Continual` base); `TimeSegments` (consecutive forecasting tasks cut by time, with the same replay); `Online` (evaluation-time stream updates after targets are observed, optionally replay). Worker batching, optimizers and local fitting are shared |
| `replay/` | replay buffers for `Online`, `ClassIncremental` and `TimeSegments`: admission, eviction and sampling; `Reservoir` (uniform over the stream, bounded capacity) |
| `adaptation/` | test-time adaptation methods for `Adapted`: each gives the parameters it updates and an unlabeled objective; `TENT` (normalization affine, prediction entropy) |
| `runtime/` | orchestrator, worker, state, seeds; `executors/Serial` and `executors/Ray` |
| `analysis/` | one table across runs: `python -m kairots report [ROOT]`; grouped mean ± std over completed runs |

Each worker retains at most eight recently used client arrays; evicted values and
split metadata can be reconstructed from the prepared store. Ray sends one small
`ClientRecord` object reference per client to each actor, while the array stays in
the prepared store until that actor loads it.

### Loss landscape

`python -m kairots landscape RUN_DIR` rebuilds a saved run from its resolved config,
prepared store and `final.pt` checkpoint. It measures train-window loss by default;
`--checkpoint best_val` selects the best-validation checkpoint, while
`--phase validation` selects validation windows. Test windows require `--allow-test`.
The analysis writes `landscape.npz` (the coordinate grid, pooled surface and one
surface per client) and `landscape.json` (configuration, center loss, pooled and
per-client Hessian sharpness estimates) under `RUN_DIR/analysis/`. A run with
client parameters (`final_clients.pt`: Ditto, LocalOnly) adds each client's loss
along the line from the global model to its own (`interpolation_client_<id>`). With the `analysis` extra installed, it also writes a 1D line or a 2D
contour (`landscape.png`); a two-direction grid additionally gets a 3D surface
(`landscape_3d.png`), with loss as height. The 2D and 3D views reuse one grid.
`analysis/landscape/one_d.py`, `two_d.py` and `three_d.py` own each view;
`common.py` shares run loading, sampling, loss and Hessian calculations.
No extra training logs are needed: checkpoints, resolved config and prepared data
are sufficient. A training trajectory would additionally need intermediate model
checkpoints, which this analysis does not currently collect. `--points`, `--span`, `--dimensions`,
`--max-windows`, `--iterations`, `--samples` and `--seed` control the calculation.

## Data shapes

| Kind | Stored rows | Values in memory | Window |
|---|---|---|---|
| Series | `date` + channels (+ `fold`, `label`) | `[steps, channels (+ label)]` | `(start, target)` |
| Panel | `sample, step, fold, label` + channels | `[samples, steps, channels]` | `(sample, label)` |

Split windows are stored as inclusive `[first, last]` ranges in
`splits/<split_key>/summary.json`.

## Contracts

- `Message` payload bills tensor `nbytes` plus four bytes per scalar, or its
  `encoded_bytes` when a codec compressed it (the tensors then hold decoded values);
  job state is not billed.
- Client data is cached once per worker; Ray actors live for the whole run and the
  round broadcast is put once per round.
- Seeds and logical job IDs stabilize random draws and scheduling. Exact repeat
  or Serial/Ray parity is only expected when the selected backend operations are
  deterministic; current parity checks pin CPU thread counts. CPU and CUDA are
  not promised bitwise equality because their floating-point kernels and
  reduction orders differ. Cross-device comparisons use task metrics and
  explicit numeric tolerances; the resolved run config records the device
  request and resource reservation.
- With a validation split, the best checkpoint minimizes the validation loss of
  what clients are scored with (`personalization_val_pooled_<loss>` when reported,
  else `generalization_val_pooled_<loss>`; the client mean with client groups);
  without one, `best_val_round` is null and the final round is the result.
- A non-finite target is an unscored position (imputation masks); every loss and
  metric skips it.
- Each client `Result.timing` records window/data preparation, fit compute, and
  total worker seconds. `rounds.jsonl` reports data, fit, evaluation, aggregation,
  transport and wall seconds. Serial client timings are summed; Ray uses each
  phase's maximum client duration. Transport is the executor-time residual after
  worker time; it includes queueing and scheduling, so it is not a pure network
  measurement.
- `Worker.update` rejects any training schedule that selects windows outside that
  client's train split. Evaluation receives one named phase at a time; adaptation
  predictions receive inputs only, and prequential target observation follows
  prediction.


## Conditional diffusion

`Denoising` fits `model.training_loss(batch_x, target)`, while Frozen scoring calls
`model(batch_x)` without targets. CSDI reconstructs the native observation mask
from prepared conditional masks and training targets, then draws self-supervised
masks. Historical masks come exclusively from the current training batches.
Known test patterns contain masks, never held-out values. Optional elapsed-hour
covariates preserve irregular timestamp spacing.

TimeGrad's recurrent features use positive lags only. Training teacher-forces prior
prediction steps; sampling appends each generated joint vector to its own history.
The sample axis is preserved through inverse scaling, marginal CRPS and CRPSSum.
CSDI's normalized CRPS uses nineteen quantiles and the median supplies point MAE.
`CompleteWindowRatio.stride` spaces eligible windows independently within each
split; target periods remain disjoint.
