# Design notes

## One-shot ridge statistics

MAE supports both loss and metric roles through the same masked absolute-error
sum. On finite targets its batch loss and gradient match PyTorch L1 loss; metric
accumulation remains separate from fitting, so a loss call retains no batch graph.

FedRidge implements the non-private fixed-regularization protocol of
arXiv:2601.08216v1, Algorithm 1, and its shared random projection extension.
Clients upload the packed symmetric Gram and feature-target moment, never a
locally fitted model. The server sums statistics and applies regularization once,
solves with Cholesky, and broadcasts the fitted weights. The fixed zero intercept
is known locally. Fitting always minimizes squared error plus ridge; a chosen MAE
evaluation score does not change that objective. A generic post-aggregation
`broadcast` hook accounts for this
message after the solution exists; ordinary iterative strategies keep their
existing pre-update downlink. A sufficient-statistic client has no local training
loss to log, so an absent value is recorded as null.

Channels are observations of the same temporal linear map, with each horizon an
independent response. Nonlinear models, covariate-packed tasks, augmentation and
continual training are refused. Statistics, the wire and the solve use float32;
the paper states no precision. The upload is the Gram's upper triangle plus the
moment, d(d+1)/2 + d values per output (Theorem 4), and the server solves by
Cholesky (Remark 5). The broadcast model is the weight alone: the paper's model
has no intercept, and the Linear bias stays fixed at zero. Above 1000
features the shared Gaussian projection uses 40% of the original dimension; this
is approximate relative to unprojected ridge. Private fusion, cross-validation and
kernel features are separate paper extensions outside this implementation.

## Informer source choices

Informer follows Zhou et al., *Informer: Beyond Efficient Transformer for Long
Sequence Time-Series Forecasting*, arXiv:2012.07436v3. Its full-history three-layer
encoder and last-quarter one-layer replica have independent weights. Two decoder
layers use masked ProbSparse self-attention and full cross-attention. Timestamp
embeddings are learned categorical tables. These rules differ from the single
encoder, continuous calendar projection and sparse cross-attention in the TSLib port.

The printed positional formula's index range exceeds the embedding width. The
cited Transformer and original author implementation define the selected encoding:
window-local positions, base 10,000 and zero-based sine/cosine pairs. Heads are
concatenated at each position, as the cited Transformer specifies; the historical
author revision `257967ed9cdef9b4139476d9cb7fdb3f41525e97` directly reshapes a
head-first context and mixes positions. The dense-attention oracle guards this.
Distillation convolution preserves length before stride-two pooling, matching the
paper's halving rule; that historical revision's padding expands length instead.
Replica cropping follows the paper's last quarter rather than the historical
stack's enumerated last-half crop. Unsampled dot products are zero in the paper's
sparsity measure, including the all-negative sampled-score case.

Omitted projection biases, initialization, calendar vocabulary and normalization
settings use the pinned original author implementation. Sparse sampling remains
stochastic during evaluation. Comparisons that vary future values or marks reset
the sampling seed to isolate the changed input.

The native 96-to-96 ETTh1 sanity case is distinct from a published-result
reproduction: the paper's ETTh1 tables have no horizon-96 cell, and its main and
appendix split descriptions disagree. A runnable adapter and equation tests alone
do not establish agreement with a paper's reported results.

## Readable setting names

Window lengths use `*_length`; DLinear uses `moving_average_kernel`; TimesNet uses
`encoder_layers`, `num_inception_kernels` and its paper symbol `k`; model feature
dimensions use `*_dimension`. Config files and CLI flags use these same names.
Old abbreviated keys are rejected with the usual unknown-field diagnostic.
Parameter and buffer keys remain unchanged so saved weights retain their format.
Internal round, operation and loss-function bindings also use full words.
Persisted result/resume keys such as `current_iter` and `accepted_ops` retain their
existing spelling; this naming pass does not rewrite stored runs.

## Explicit missing-value handling

Standard scaling rejects missing observations before split filtering and fits only
training rows. This prevents a missing validation or test value from disappearing
through window exclusion. Masked handling is an explicit choice: it preserves NaNs
and the previous missing-aware statistics for mask-reading models. Irregular sources
declare this requirement, so omission fills the mode and explicit conflicts fail.

Design choices and differences from published papers and original repositories.
TSLib below means the original [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
repository at `4e938a1767106324dd753b2a44832bf870a0252e`; it is a source for its own
task protocols and model adaptations, while each model's original paper remains
the authority for paper reproduction.

## Training-window selection

Ordinary `Offline` training selects the full training split on each update.
`ClassIncremental` selects released label-group examples, samples replay before
fitting and admits newly released examples afterwards. Replay state remains in the
client checkpoint. The non-selectable `Continual` base owns period releases and
replay; `ClassIncremental` supplies classification-label grouping and constraints.
The worker calls the schedule's `select` and `commit` methods while
keeping batching, optimizer handling and strategy-specific fitting shared.
Class-period evaluation and release-boundary scaling apply only to schedules
declaring `CONTINUAL`; explicit `Offline` works for any task without
classification-only attributes or reports.


## Class-count allocation

ClassPerClient implements the original PFLlib pat branch at
`0169ba7e412c9856a08bb3faefab1e35f538a3c1`, cited as PFLlib's own allocation
(Zhang et al., JMLR2025, arXiv:2312.04992v2). It does not implement McMahan shards.
The first eligible clients receive each class. Equal-size splits truncate and give
the last selected client the remainder; unbalanced splits use original bounds
`int(max(num_per / 10, least_samples / num_classes))` to `int(num_per)` exclusive,
where `least_samples = int(min(40, N / num_clients / 2))`. The40 comes from original
batch10 and train ratio0.75, and does not change the experiment's evaluation split.
An isolated Generator supplies random draws; the original uses global randomness.

Some client/class counts exhaust all class slots before every label is covered.
The original silently stops and drops uncovered samples. Preparation instead fails
on that case, empty clients, invalid random-size ranges and missing sample IDs.
It does not rebalance or change the requested allocation to conceal invalid data.


## Pathological shards

Pathological implements McMahan et al.'s label-sorted equal-shard partition
(arXiv:1602.05629v4, Section3). Its paper setting of100 clients, two shards each
and60000 MNIST samples gives300 examples per shard. Other sizes must still form
equal nonempty shards, or preparation fails. No samples are dropped or repeated.
Ties within a label retain source order, an ordering the paper leaves unconstrained.
An isolated seeded Generator permutes shard IDs. This is different from PFLlib's
pat branch, which assigns classes and splits their examples between eligible clients.


## Extended Dirichlet allocation

ExtendedDirichlet implements the two-level label partition in Li and Lyu,
NeurIPS2023 (arXiv:2311.03154v3), rather than their SFL training algorithm.
Paper omissions follow the original author implementation at
`5972d5c0692c0f8600711e37630b73ba09efe306`, `sim/data/partition.py`:
coverage divisor5, balancing below mean client size, last eligible client rounding
remainder and minimum10 samples with unbounded retries. PFLlib uses divisor2;
that change is not adopted. An isolated default_rng replaces global randomness.

Impossible coverage and minimum sizes, conflicting labels and numerically vanished
proportions raise errors. One client receives all samples without indexing empty
cut points. Materialized rows use sample/step order required by the store rather
than the author's final shuffle. Direct comparisons inject the same Generator into
the original function and compare client sample memberships. Shared labeled
operations are a base class, so their source participates in prepared-data keys.


## Objective and score roles

The `metrics/` category contains batch objectives and mergeable scores. `ROLES`
controls whether a component may be selected as a loss or a metric. `GaussianNLL`
owns both roles; its metric result key is `gaussiannll`, replacing `nll`. `MSE`,
`SampleCRPS` and `Return` also support both roles. Calling an objective does not
change its metric accumulators.

## Optimizer components

`learner.optimizer` and `Adapted`'s `evaluation.optimizer` are nested component
configs. Adam retains zero weight decay; SGD retains zero momentum and weight
decay by default. Learning rates stay with their caller, and fused construction
is requested only when the PyTorch optimizer supports it. Ditto's personal SGD
and test-time adaptation retain their unfused construction.

Optimizer-name strings are not accepted. Saved configs using strings need the
equivalent nested object before they can resolve; optimizer tensor-state formats
are unchanged.

## Continual preprocessing release boundary

The scaler fits on the first round's released examples only. Training sample
indices are sorted before release, matching the worker's window order. The scaler
stays fixed for later rounds and tasks so model coordinates never change.

## TimesNet task identity and positive frequencies

`reconstruct` follows the task's reconstruction contract. Forecasting always
extends the time axis, including equal input and output lengths. Reconstruction
does not extend it. Existing equal-length forecasting checkpoints lack
`head.predict_linear.*` and need retraining or explicit initialization of that
layer; checkpoint parameter names for existing layers are unchanged.

FFT period selection excludes DC by selecting only positive-frequency bins.
`k` must be between one and half the transformed sequence length. This is a
deliberate correction to TSLib's zero-amplitude DC selection, which can still pick
index zero on short or flat inputs. Cases that select the same positive bins keep
the same computation.

For each selected frequency `f`, TimesNet paper Equation 1 specifies period
`ceil(T / f)`. The pinned TSLib implementation uses floor division instead.
KairoTS follows the paper and pads to the paper period before reshaping. Therefore
fixed-weight numerical comparisons against TSLib are not valid when `T / f` is
nonintegral; its task output shapes remain the reference contract. A nondivisible
length/frequency fixture checks the paper's padded reshape directly.

## R2 channels with missing observations

R2 averages channels with at least one finite target. Entirely unscored channels
are omitted, and no scored channels is an error. A scored constant-target channel
makes the aggregate NaN, including perfect constant predictions: its denominator
is zero. Reports omit non-finite values and retain their finite observation count.

## Fine-tuning and FedBABU

With `evaluation.finetune_epochs > 0`, each seen client fine-tunes every parameter
of a copy of the model it is scored with, on its own train windows, after the final
round; the copy is scored as `finetuned_*` and discarded. Fine-tuning traffic is
not billed, unseen clients are not
fine-tuned, a best-validation round has no `finetuned_*` scores, and `client_model`
is refused.

`FedBABU` freezes the head during the rounds and sends and averages only the body,
so its billed traffic is the body. Its source is Oh et al., "FedBABU: Toward Enhanced
Representation for Federated Image Classification" (ICLR 2022,
[arXiv:2106.06042v3](https://arxiv.org/abs/2106.06042v3)), with original code at
[jhoon-oh/FedBABU](https://github.com/jhoon-oh/FedBABU). The framework's evaluation
fine-tunes the whole model; a paper reproduction must also match the selected
fine-tuning scope and budget. FedBABU needs a head with parameters, which DLinear's
forecasting head lacks.

## FML is classification-only

FML distills class probabilities with KL over the class axis, so configuration
refuses forecasting. The meme optimizer's per-parameter state restarts each round
and keeps its scheduled learning rate; the private optimizer persists and takes that
learning rate. A
client without a `client_model` starts its private model from the server model.
Every client trains every round.

## SCAFFOLD keeps a server control variate

`SCAFFOLD` follows Algorithm 1 of Karimireddy et al. (2020): unweighted means over the
selected clients, `global_step_size` for the model step and `sum(c_i+ - c_i) / N` for
the control step. The server control variate lives on the
strategy and is saved with the resume checkpoint; each client's lives in
`ClientState.extra["control"]`. Both travel as `control.`-prefixed tensors, so they
are billed in each direction. Per-client downlink tensors that are not the broadcast
model travel with the Ray job. The local steps add `c - c_i` to every parameter's
gradient through a hook and need plain SGD (`learner.optimizer` `SGD`); option `I`
takes the gradient at the server model as the mean of the per-batch gradients over one
pass. Control variates cover parameters, so a model with buffers (batch norm) is
refused. Option `II` divides the local displacement by the summed step sizes of the
update, which is `K * lr` under a constant rate and stays finite under a schedule that
restarts each round.

## APFL mixes parameters

`APFL` follows Algorithm 1 and eq. alpha_gd of Deng et al. (2020). Each batch steps
`w` and `v` from the parameters before the step, and `v`'s gradient is taken through
the mixed parameters `alpha * v + (1 - alpha) * w`, so it carries the factor `alpha`.
The official FedTorch code differs in three places, and the paper's formulas are kept:
it mixes model outputs rather than parameters, adds `0.02 * alpha` to the `alpha`
gradient, and computes that gradient from `alpha * grad(v) + (1 - alpha) * grad(w)`.
From that code `APFL` takes what the paper leaves open: `alpha` moves once per round, on
the first local step, is clipped to [0, 1], and `v` starts from the global model. The
global model steps with `learner.optimizer`; `v` and `alpha` take plain SGD steps at the
learner's learning rate. The paper's 1% learning-rate decay per step is not applied.
The personal state lives in `ClientState.extra` (`personal`, `alpha`); a model with
buffers is refused.

## Loss landscape pools scored positions

Loss surface and Hessian calculations weight each batch loss by its finite target
count, then divide by the total count. This matches `Frozen` evaluation when batches
contain different numbers of scored values, including batches with masked targets.

## Ditto and personal models

`Personalized` keeps each client's personal model in
`ClientState.extra["personal"]`; `client_parameters` makes evaluation score it.
`Ditto` uses personal SGD with `mu * (personal - global)` (written as the
loss term `mu / 2 * ||personal - global||^2`), then the usual global update. The
objective comes from Li et al., "Ditto: Fair and Robust Federated Learning Through
Personalization" (ICML 2021,
[arXiv:2012.04221v3](https://arxiv.org/abs/2012.04221v3), Section 3 and Algorithm 1).
The current personal-before-global execution order is a framework choice; this
citation alone does not establish full paper-protocol equivalence. Every
personal model starts from the initial global model. The `Traditional` family now
owns the sample-weighted mean; `FedAvg` only names it.
Configs list only non-default values for `model`, `learner`, `strategy` and
`runtime` so components can be swapped; `task`, `data` and `evaluation` stay fully
spelled out because they define the benchmark protocol.

## LocalOnly and Centralized; per-client parameters

`LocalOnly` is a standard baseline under `NonFederated`. It starts every
client from the same initial model, keeps it in `ClientState.model_state` and sends
nothing. `Centralized` is a standard baseline under `Traditional`. It runs one job per round on
the pooled train windows of all clients; the job is a message from the server to
itself, which bills 0 bytes. Pooled batches, each from one client, are shuffled together. Evaluation scores
each client with the global parameters overlaid by its `model_state`, which is also
the hook for personalized methods. CLI overrides now edit the file's values, so
`--strategy.name X` resolves X's own defaults.

## Anomaly detection follows TSLib; TSLib is the main reference

TSLib is the main reference for task protocols, loaders and models. Anomaly detection
ports `exp_anomaly_detection`: reconstruct windows of 100 steps, score each step by its
channel-mean squared error, set the threshold at the `100 - anomaly_ratio` percentile
of the calibration scores, report point-wise `F1` and point-adjusted `AdjustedF1`
separately. Deviations: SMD is federated (one client per machine, each calibrates
its own threshold); validation is held out from training (TSLib also trains on it);
the threshold comes from validation scores by default (`calibration="train"` uses
train scores for runs without a validation split). TSLib pools train scores with
unlabeled test scores, which makes the threshold depend on the test data, a practice
recent reviews of anomaly benchmarks count as test-set leakage; `calibration=
"train_test"` keeps it for comparison with TSLib.

`SMDAll` concatenates the 28 pinned OmniAnomaly machines in the author loader's
machine order (all training rows, then all test rows) into one series. This
supports the TimesNet paper's centralized SMD experiment; `SMDGroup1` remains
the eight-client federated example. `SMDAll-TimesNet-Historical.json` reproduces
the author loader's overlapping train/validation population and train-plus-test
threshold. `SMDAll-TimesNet-Paper.json` uses held-out training windows and the
paper's validation-only threshold. Both retain 100-step nonoverlapping windows,
the 0.5% threshold ratio, TimesNet dimensions, Adam, 10 epochs and the pinned
author schedule. The validation-only run is the leakage-free comparison; the
historical run is labeled separately because its threshold includes test scores.
With the pinned settings and seed 2021, the historical run scores AdjustedF1
0.857903 (published TimesNet SMD: 0.8581); the validation-only run scores
0.823999 and is a separate leakage-free benchmark, not a paper-number comparison.
The run uses all 28 machines (708,405 training and 708,420 test rows), with
7,084/1,416/7,084 historical train/validation/test windows and
5,667/1,416/7,084 paper-clean windows. The difference between the two protocols
is expected because their threshold populations differ. Do not merge their scores.

The source carries `fold` and per-step `label` columns; the store appends the label
as a last, unscaled column, and the evaluation protocol receives every split so it can
calibrate. Sources may declare `TASKS` (SMD: anomaly detection only). `Standard` now
keeps scale 1 on constant channels, as scikit-learn does (store format 3).

## Imputation scores hidden positions only

`Imputation` reads target windows (`input_length = 0`, `output_length = L`) and hides a
random `mask_ratio` of the observed values. The model sees the window with hidden and
natively missing values set to 0. The target holds the true value where hidden and
NaN elsewhere: every loss and metric treats a non-finite target as unscored, so one
convention covers masks without changing their signatures. Tasks declare `HEAD` (the
model head they need) and `RECONSTRUCTS` (the model input is the target window).
Scoring masks restart from the task seed for every client and evaluation.

The masked MSE divides a sum by a count instead of calling `mse_loss`; forecasting
training is unchanged (R2 identical) and the pooled MSE moved by about 1e-7 relative
(DLinear reference now 0.24636848270893097).

## Classification uses panel data

A panel source (`Panel`, e.g. `UEA/BasicMotions`) yields one row per sample and step.
`IID` deals whole samples to clients; `Predefined` keeps the archive's official
train/test folds and draws a stratified validation set from train. The store keeps
panel clients as `[samples, steps, channels]`, and a window is `(sample, label)`.
The task supplies the default loss and metrics (`Classification`: `CrossEntropy`,
`Accuracy`). Model shape fields (`input_length`, `num_classes`, `head`, ...) are filled
from the data only when the model's `Config` declares them. The best checkpoint
minimizes the validation loss (`<scoring>_val_pooled_<loss>`), so it follows the
task's loss.

UEA's own site rejects scripted downloads (HTTP 401); the aeon and sktime repositories
ship byte-identical copies, pinned by sha256.

## Where repository rules live

Repository rules live where they are used: `docs/contributing.md` and
`docs/reference/conventions.md` for people, `scripts/check_style.py` for the checks,
`README.ai.md` for the code map.

## Raw data is downloaded into the repo

Each source lists its raw files as `RemoteFile`s (path, mirrors, sha256, optional
archive extraction); `data/download.py` is the only download code. Files land in
`datasets/<Family>/raw/` and are never modified; prepared data goes to
`datasets/<Family>/prepared/<data_key>/`. Split windows are stored as `[first, last]`
start ranges in `summary.json`, because positions are derivable from the config.

## Naming rules are written down and checked

`reference/conventions.md` holds the rules; `scripts/check_style.py` enforces them
(N2: a PascalCase file defines its class, so `composite.py` became `Composite.py`;
W1: one spelling per word, e.g. "directory").

## Metrics declare where they apply

Metrics and losses declare `TASKS` and `NEEDS`; metrics also declare `DIRECTION`, and
may declare `PHASES`. `Experiment` rejects an incompatible choice before any data is
prepared. Evaluation selects metrics per phase
(`evaluation.metrics = {"validation": [...], "test": [...]}`); `"train"` scores the
frozen global model on train windows.

R2 is test-only (`PHASES = ("test",)`), so it is reported but never drives model
selection or checkpointing.

## Prepared data lives in a Parquet store

Raw files are read once into `datasets/<Source>/<data_key>/`; splits and scalers are
cached under `splits/<split_key>/`. Both keys hash their inputs, so a changed config or
raw file gives a new directory instead of rewriting one. Workers receive a small
`ClientRecord` and read only the clients they train.

## Gossip runs on the driver after local training

After local training, `Strategy.exchange` mixes the returned client models once (Metropolis weights)
before they are stored. Each directed edge is one billed `Message` carrying the
sender's full model, reported as `peer_bytes`; downlink and uplink stay 0. The topology
is built over the selected clients only.

## Families combine; clashes fail at import

KairoTS lists the hooks a family may own
(`strategies/base.HOOKS`); a class whose parent families both override one fails at
import unless it defines that hook itself. The local objective is its own family,
`Proximal`, so FedProx and DFedProx share it. FedProx anchors on the downlinked global
model; KairoTS anchors on the model the client starts the round from, which is the same
for FedProx and the client's mixed model for DFedProx. `DFedProx` is a framework
composition of gossip and the FedProx penalty, rather than a standalone paper
benchmark. The penalty's original source is Li et al., "Federated Optimization in
Heterogeneous Networks" (MLSys 2020,
[arXiv:1812.06127](https://arxiv.org/abs/1812.06127)).

## Naive is a model, not a learner

TSLib's `Naive2`/persistence baselines live beside the models; KairoTS keeps `Naive` a
parameter-free model (`TASKS = ("Forecasting",)`, now checked like sources and
metrics) and pairs it with the `Stateless` learner (no optimizer, 0 epochs), so the
round flow, evaluation and artifacts are unchanged and 0 bytes are billed.

## Augmentation is per batch, not once at load

TSLib applies `run_augmentation_single` once to the training array when the dataset is
built, so the same noisy copy is reused every epoch. KairoTS applies the augmentation
to each training batch in the worker with the job's seeded generator: fresh noise per
round, deterministic per seed, identical on Serial and Ray. Inputs only, as TSLib
(`data_x`, not `data_y`). Prepared data and store keys do not depend on it.

## Schedulers span the run or each round

A schedule spans the run by default (`span` "run"): each client's scheduler steps
after its local update and its state travels in `ClientState.scheduler_state`.
A client that is not selected does not step, so under partial participation it lags
the global round count. With `span` "round" the schedule restarts at every client
update and steps after each optimizer step, across that update's epochs and batches;
no scheduler state is kept. A strategy's extra optimizers, such as Ditto's and FML's
personal optimizers, are not scheduled within the round.

## Initialization policies run once, on the server model

`Random` keeps PyTorch's layer initialization. Other policies run once on the
canonical global model, immediately after construction under `runtime.seed`.
For example, `Average` sets selected body-weight patterns to `1 / input_length`.
Clients receive those shared values, so Ray workers do not re-randomize them.
Policies draw from the global torch RNG after construction; the manifest records
the policy, seed and each parameter it set.

## Held-out clients are their own population

`Grouped` keeps the ratio split for every client and holds out a seeded share of
clients (at least one, never all). Held-out clients never train, are never selected
and are scored on test only, as `unseen_*` metrics; pooled `test_*` metrics cover seen
clients only, so they are not comparable with a run without held-out clients. Each
held-out client is still scaled on its own early (train) portion, as the store fits
scalers per client; a strict zero-shot protocol would reuse seen-client statistics.

## Quantile levels come from the output shape

Probabilistic forecasting starts with a quantile head (`[B, H, C, Q]`) and evenly
spaced levels `(k + 1) / (Q + 1)`, so the task's `num_quantiles` is the one setting
and the loss and metrics read `Q` from the prediction rather than repeating a level
list. CRPS is the quantile approximation (twice the mean pinball loss over levels),
not an analytic or sample CRPS; with `Q = 1` it equals MAE. Scores are on the scaled
values, like every other metric so far. Quantiles may cross (no monotonicity
constraint), and no point projection is saved, so point metrics (MAE) are not offered
for this task. Calibration is scored on the outer central interval (first and last
level, nominal `(Q - 1) / (Q + 1)`): `CoverageError` is the gap between its empirical
and nominal coverage, `IntervalWidth` (test only) its mean width.

## XGBoost is one multi-output booster, never averaged

The XGBoost model maps the flattened input window (`input_length * input_channels`) to
the flattened forecast with one multi-output booster (xgboost's default one output
per tree), continuing from the previous booster each fit, so every round adds
`num_trees` rounds. Its state is the booster's UBJ bytes in a uint8 buffer, which the
runtime carries like any tensor; parameter averaging would corrupt it, so the model
declares `STRATEGIES = ("LocalOnly", "Centralized")` and the schema rejects the rest.
Federated tree methods (histogram sharing, SecureBoost) are later strategy work.
The training loss is xgboost's squared error, not the configured torch loss.

## Partial training starts with masked updates on whole groups

FedPMT's original source is Wu et al., "Straggler-resilient Federated Learning:
Tackling Computation Heterogeneity with Layer-wise Partial Model Training in Mobile
Edge Network" ([arXiv:2311.10002v1](https://arxiv.org/abs/2311.10002v1), Section III
and Algorithm 1). It retains a full forward pass and restricts backward updates to
a suffix of layers. KairoTS freezes whole parameter groups (one module's own
parameters) with `requires_grad`, so the optimizer skips them. Clients upload
trained parameters; the server takes the unweighted mean over clients that trained
each parameter. Client-index depth assignment is the framework's allocation policy;
the citation does not establish full paper experimental equivalence.

## Codecs carry decoded values and bill the encoded size

`Message.encoded_bytes` records a computed upload size: the tensors hold
what the server would reconstruct, and billing uses the codec's wire size, so no bit
packing is simulated. FedPAQ's stochastic rounding draws from a generator seeded with
the job seed, so Serial and Ray runs agree. The method source is Reisizadeh et al.,
"FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging
and Quantization" (AISTATS 2020,
[arXiv:1909.13014v4](https://arxiv.org/abs/1909.13014v4)). Codec checks alone do not
establish the paper's complete training and communication protocol.

## The run report groups by config hash

`python -m kairots report` reads every `runs/*/` directory and prints one row per
task, source, model, strategy and config, with `mean +/- std` over runs. The config
column is the first 8 characters of a sha256 over the resolved settings without the
`runtime` namespace, so runs that differ only in seed, device or executor share a row
while any other setting change starts a new one. Cells use ASCII `+/-` because the
Windows console garbles `±`. One module (`analysis/report.py`) replaces the planned
io/aggregate/report split; split it when plots arrive.

## M4: one client per series, validation held out, MSE loss

M4 Hourly has 414 univariate series of different lengths, so each series is one
client (`Collection` kind, `PerFile`); a client tensor stays one series and windows
never cross series. `TestFold` forecasts the official 48-step test fold from the
history before it, as TSLib's `last_insample_window` does, but validation is the last
48 training steps rather than the test set TSLib validates on. Train windows step by
`train_stride` (24 in the config) instead of TSLib's random cut points within the
last `history_size * pred_len` steps, and the loss is the default MSE, not TSLib's
SMAPE loss. Values are not scaled (`Identity`), as in TSLib, because SMAPE is scored
on the values the model sees. MASE takes each window's whole history (for the test
window, the full training series, as `m4_summary.mase` does) through Frozen's
`history` argument; it is unchanged by a per-channel affine scaler. Check: the Naive
model reproduces the M4 paper's Naive1 Hourly SMAPE 43.003 and MASE 11.608.

## TimesNet without time marks or imputation masks

KairoTS models take only `batch_x`, so TimesNet's embedding is value plus position;
TSLib's time-feature embedding is left out (its weights would be averaged and never
trained). Reconstruction tasks (imputation, anomaly detection) use the forecasting
head with `input_length == output_length`, which is TSLib's `anomaly_detection` path. For
imputation TSLib normalizes each window with the mean and spread of observed
positions only; here the model gets no mask, so the zeroed hidden positions count.
Classification uses a padding mask of ones (equal-length samples). Task-path shape
checks against TSLib do not establish numerical parity, masked imputation fidelity
or paper benchmark fidelity.

Series Stationarization uses each window's per-channel population variance, with
`1e-5` added before taking its square root. De-normalization scales the prediction
and then adds the input mean. This follows Algorithm 2 of Non-stationary Transformers
and its corrected arXiv v4 Equation 2; the original 2022 Equation 2 incorrectly puts
the mean inside the product. TimesNet paper Equation 1 specifies
`ceil(T / frequency)` for each period; the pinned official implementation floors it.
KairoTS follows the explicit paper equation. For lengths where the quotient is not
integral, the padded 2D reshape can differ from that author-code version. The
fixed-weight TSLib comparison uses a length whose selected frequencies divide it,
and a separate length-10/frequency-3 fixture checks the paper's ceiling behavior.
Calendar embedding, imputation masks and other paper details remain implementation gaps.

## PeMS08 is one network frame; PerNode makes the clients

KairoTS keeps the network as one long `date, node` frame (the `SpatioTemporal` kind) and stores the
road graph (`PEMS08.csv`, 295 edges) in the manifest, so the partition decides the
clients: `PerNode` gives one per sensor, and a later graph model can read the whole
network. The config keeps sensors 0 to 7 so smoke runs stay short; `nodes: []`
selects all 170. No model reads the graph yet.

## TENT adapts LayerNorm, keeps eval mode, scores pre-update

TENT (Wang et al., ICLR 2021) adapts BatchNorm affine parameters with test-batch
statistics in train mode. KairoTS models have no BatchNorm, so `TENT` adapts
LayerNorm and GroupNorm affine parameters and keeps the model in eval mode (dropout
off). This is a partial implementation: BatchNorm support and the paper's offline
adapt-then-score protocol are missing. Each batch returns the prediction from the
forward pass of its last update step. With one update per batch, the current
LayerNorm/GroupNorm path's first batch matches frozen evaluation; canonical BatchNorm TENT recomputes target
statistics before predicting, so its first batch need not match frozen evaluation.
`Adapted` adapts every test evaluation (each round), starting each client from its
own parameters, and restores them afterwards.

## Jena weather drops one repeated date and keeps TSLib's error codes

TSLib reads `weather.csv` by row position. KairoTS windows by date, and the file
repeats 2020-05-12 06:00, which made the inferred step zero and every window
invalid; the source keeps the first of the two rows. The 1 h 40 min gap on
2020-05-29 stays a gap, so windows across it are skipped (TSLib joins them). The
-9999 wind error codes are kept, as in TSLib, so results stay comparable.

## RowConcat and ColumnConcat are composite sources, not a composition package

The design sketch places concatenation under `data/composition/`. Sources already
compose (`Composite`, `ETDatasetHour`), and the store, cache key and schema resolve
one `data.source`, so both are sources that wrap one member source. `RowConcat`
turns each channel into a univariate series and client (one household per client
for an Electricity-style set); `ColumnConcat` turns each client into channels of one
multivariate client, inner-joined on date with no fill. Because model width comes
from the config, `ColumnConcat` needs its channel names listed in `input_channels`.
The multivariate file is the original source, and `RowConcat` gives the
per-household univariate split with one client each.

## FeatureJoin joins on exact timestamps only

The design sketch places `FeatureJoin` under `data/composition/` with declared
entity keys and availability rules. It is a composite source like `ColumnConcat`,
and its only key is `date`: each member must have one client, its declared channels
become `<member>/<channel>`, and members are inner-joined on exact date matches.
Unmatched dates are dropped, a duplicate date raises, and no nearest-time or as-of
matching is done, so a row never reads an observation from a later timestamp.
Release delays (a feature published after its timestamp) are not modelled.

## Prequential runs on the test phase, batch-causal

TSLib has no online protocol. `Prequential` walks each client's test windows in time
order after the frozen pass. A batch is predicted at once, so updates before it use
only windows whose targets end by the batch's earliest origin: windows inside a
batch never learn from each other, and with batch size B and stride 1 a window's
targets wait for the next batch whose origin passes them. Each client restarts
from its own parameters with a fresh learner optimizer, and the model is restored
after the test phase, so federated training never sees test targets. Online updates
use the experiment learner's optimizer and learning rate, `steps` per release.

## Online optionally replays the test stream and measures retention on validation

TSLib has no continual protocol. `Online` runs inside `Prequential`, so its
stream is the test stream: the replay buffer holds only released test windows,
never train or validation windows, and restarts with each client. Retention is
the client's end-of-stream model scored on its validation windows, the period just
before the stream, against the frozen model; there are no task boundaries, so
there is no per-task forgetting matrix. Replay sampling uses the buffer's own seed,
not the experiment seed.

The Online and OnlineReplay configs cut each client 20 / 5 / 75 (warm-up train, validation,
stream), the online-forecasting split of FSNet (Pham et al., 2023; its
`Dataset_Custom`), so most of the series is streamed; `--data.split.ratios`
changes it. FSNet's ETTh loader keeps only the first 20 months (4 / 1 / 15
months); KairoTS cuts the whole file by ratio.

## Leakage audit: what training may and may not see

`scripts/leakage.py` trains twice, once with the test set poisoned, and requires
bitwise-identical train/validation numbers and parameters. Scalers fit on train rows
(series: up to the last train target; panels: train samples). Train and validation
windows end before the next cut, and the best checkpoint is picked on validation.
Test scores are logged every round but never select anything. `Adapted` and
`Prequential` touch test data only inside the test phase and restore the model.
One known exception, outside training: `Calibrated` with
`calibration="train_test"` (TSLib's anomaly protocol, not the default) sets the
threshold from unlabeled test scores.

`CompleteWindowRatio` places its per-client cuts at a share of the rows, so the
borders do not move with the input or output length and missing values cannot
move them. For hourly ETTh at 0.8 the train/test border is row 13936: train uses
rows 0 to 13935, and test starts at row 13936 and runs to the last row, which
gives 3293 test windows for 96-to-96.

## Forecasting configuration and scoring

The forecasting examples use `FedAvg` and `Frozen`: Adam at 1e-4, batch 32,
one shuffled local epoch, persistent client optimizer and scheduler state, and
train-window-weighted model aggregation. The aggregated model is evaluated after
each round. These are example settings, not a paper benchmark recipe.

Reports select the final round, or the best validation round when validation is
configured. Every scored value has equal weight in pooled MSE;
`generalization_test_mean_client_mse` instead averages the per-client MSEs.
`Standard` uses the population standard deviation and keeps scale 1 for constant
channels. Partial participation selects `ceil(n * join_ratio)` clients.
`CompleteWindowRatio` permits test lookbacks into earlier splits by default;
the forecasting examples set `lookback_crosses_boundary` false.

## Generalization and personalization scores

Generalization (`generalization_*`)
loads the server model into every client after aggregation. A strategy whose clients
keep parameters of their own (Ditto's and APFL's personal models, local layers, a
`client_model`) is also scored with each client's own parameters, as personalization
(`personalization_*`). Traditional methods report generalization only, since a
client's model is the server model. Keys read
`<population><scoring>_<cohort><phase>_<pooled|mean_client>_<metric>`, for example
`unseen_generalization_test_pooled_mse`.

## Client groups and cohorts

`groups` lists client groups, each overriding the experiment's `data` components and
`client_model` and keeping `clients` (partition client ids, all when empty).
A group component
with the base component's name (or none) overrides its fields; another name replaces
it with its own defaults. Each group prepares its own clients, cuts and train-fitted
scalers; run client ids number the kept clients in group order, and the manifest
keeps each one's `partition_client_id`. The server model (`model`) is resolved on
each group's data and must have the same parameter shapes in every group, so a model
with per-channel weights fails across groups with different channel counts.
`client_model` is the model a client trains and is scored with when a strategy that
declares `CLIENT_MODEL` shares another one.

Clients whose scoring models have the same parameter names and shapes form a cohort
(one architecture at one width and depth), listed in the manifest under `cohorts`.
Scores cover every client, and each cohort alone (`personalization_cohort0_test_*`)
when there are several. Pooled scores across groups mix units when
`evaluation.inverse` is on; the scaled default keeps them comparable. Groups and
`client_model` are scored by the `Frozen` protocol only.

## test scores are scaled by default

Test metrics are scored on scaled values by default. With
`evaluation.inverse` true (TSLib `--inverse`), each client's train-fitted scaler is
undone on predictions, targets and the MASE history before test metrics; raw errors
show how a model handles a channel with a large range, which scaled errors hide.
Validation stays scaled, so the best-checkpoint choice is the same either way.
Classification labels are left alone. `BottomUp` turns inverse on by default, since
reconciliation sums raw values. ETDatasetHour DLinear 2 rounds gives the scaled test
MSE 0.2682070732116699 and, with `--evaluation.inverse true`, 10.65915584564209.

## Validation is opt-in outside anomaly detection

TSLib splits ETT 12/4/4 months and other sets 70/10/20, and reports the test score
of the epoch with the best validation loss. KairoTS defaults to train/test 80/20
with no validation and reports the final round, so no data is held out only to pick
a round. A validation share (`ratios` [0.7, 0.1, 0.2], `validation_windows`,
`validation_ratio`) turns the best-round pick back on. Anomaly detection keeps a
validation split, because its threshold is calibrated on validation scores.
`Prequential` reports forgetting only when validation windows exist.

## Covariates: role layout, NaN horizon, TiDE

TSLib passes time marks (`x_mark`, `y_mark`) next to the values and leaves each
model to read or ignore them. KairoTS declares roles on the task instead:
`target_channels` (the rest of the source channels are past-observed covariates)
and `calendar` (TSLib's `timeF` features, `data/calendar.py`). The store appends
the calendar features, unscaled, after the scaled channels. `Forecasting.prepare`
gives the model `[targets | past | calendar]` over `input_length + output_length` steps
with the horizon values of targets and past covariates set to NaN, so a value not
observed at the forecast origin cannot reach the model. A run that asks for
covariates with a model that does not declare `COVARIATES` is rejected rather than
silently ignoring them. Inverse scoring and MASE history use the target columns
only. Release lags, forecast vintages and static covariates are not modelled.

`TiDE` follows TSLib `models/TiDE.py` with two differences: each stacked residual
block has its own weights (TSLib's `[ResBlock(...)] * n` repeats one instance, so
the stack shares weights), and the flattened history of past-observed covariates
is an extra encoder input (TSLib's TiDE has no past-only covariates). Without
calendar features the feature encoder is left out instead of reading zeros.
Defaults follow TSLib's ETTh1 script (hidden 256, decoder output 8 per step,
dropout 0.3). On ETDatasetHour OT, 3 rounds, scaled test MSE was 0.0920 with HUFL
and calendar covariates and 0.0919 on OT alone; DLinear on OT alone gave 0.0849.

## Self-supervised pretraining is two runs and an initialization policy

The design sketch has `paradigms/SelfSupervised.py` and `training/stages/`. Here the
pretext task is the existing `Imputation` task (random masked reconstruction, as in
TSLib), and fine-tuning is a second run whose `initialization` is `Pretrained`: it
copies every checkpoint tensor whose name and shape match and leaves the rest (for
TimesNet, the `predict` projection that maps `input_length` to `input_length +
output_length`) at native values. `configs/imputation/ETDatasetHour-TimesNet-Pretrain.json`
writes to `runs/pretrain`, and `configs/forecasting/ETDatasetHour-TimesNet-FineTune.json`
reads the latest run there. Both use `cut: "shared_calendar"`, so the two splits cut
at the same timestamp although their windows differ in length (a per-client cut
counts windows, so a shorter pretext window would move the cut later). `Pretrained`
refuses a pretraining run whose held-out data starts after this run's. This fine-tune
freezes no weights (`Transfer` can). Smoke runs the pretraining config first and
fine-tunes from its temporary checkpoint.

## Vertical FL is the SplitNN strategy and model, not a federation package

The design sketch has `federation/Vertical.py`. Vertical training here is the
`SplitNN` strategy paired with the `SplitNN` model (each names the other, so
FedAvg on SplitNN or SplitNN on another model fails at config time). Parties own
channels of one timeline: the label party owns the targets, and each feature party
owns a block of past-observed covariates (`party_channels`, in source order without
the targets). Rows are aligned by date because they come from one table, or from
`FeatureJoin` for parties in different files. Synchronous split learning on one
aligned client equals end-to-end training of the block-structured model, so the
strategy trains it like `Centralized` and bills, per training window and epoch,
each feature party's cut-layer activation up and its gradient down. Evaluation
activations are not billed. Vepakomma et al. run the label party's top model on
the server and add no label-party residual; here the label party also maps its own
target history linearly to the horizon, as DLinear-style models do. All parties run
in one process, so no privacy property is claimed.

## Transfer is a strategy, a split budget and the Pretrained guard

The design sketch has `federation/Transfer.py`. The transfer setting here is three
pieces: a source run (for example `ETTh1-TimesNet-Source.json`, FedAvg on the
source population, writing to `runs/source`); the target run's split
`train_windows`, a few-shot budget counted in windows per client and taken as the
last train windows before the cut; and the `Transfer` strategy, where each target
client fine-tunes the `Pretrained` model alone with `frozen` parameter patterns
held fixed. FedHealth (Chen et al., 2020) freezes the convolution layers and
retrains the fully connected ones, which `ETTh2-TimesNet-Transfer.json` mirrors by
freezing the TimesNet embedding and blocks; its CORAL alignment between source and
target features is not implemented. The scaler still fits on every target train
row up to the cut, not only the budgeted windows. `Pretrained` holds a source
client that the target run lacks to the target's earliest held-out timestamp,
because ETTh1 and ETTh2 share a calendar and a later source window could carry
information about the target's test period. Source and target share channels, so
no channel correspondence is mapped.


## TourismLarge carries the summing matrix; BottomUp reconciles

`TourismLarge` reads Nixtla's copy of the Australian domestic tourism hierarchy
(`data.csv` and its summing matrix `agg_mat.csv`) and refuses data whose series do
not equal the sums of their bottom members. The manifest keeps the summing matrix
as each channel's bottom-level members. A model forecasts all 555 channels
independently; under `Frozen` the forecasts need not add up. The `BottomUp`
protocol keeps the bottom-level forecasts, sums them in raw units through the
matrix and scores every level, so the forecasts add up. Bottom-up still trains on
every series; MinT and other optimal-combination methods are not implemented. The split is the usual ratio split on 228 months, not Nixtla's
last-12-months test.

## Irregular is a Panel with padded observation rows

The design sketch has `Source -> Irregular`. `Irregular` subclasses `Panel` instead,
so the store, splits and views that handle panels handle it unchanged: each row is
one observation time of a sample, `time` is the first channel, unobserved values
are NaN and trailing padding rows are NaN everywhere. The `Standard` scaler skips
NaN, so it fits on observed train values only. `PhysioNet2012` uses set A as the
train fold and set B as the test fold, one client per ICU type, and keeps every
variable including the descriptors Age, Gender and Height as channels observed at
time 0. ICU type is the client, not a channel. Raw values are not cleaned (a few
temperatures are negative). `GRUD` reads the mask and time channel and declares
`MASKS`; the schema refuses a model without it on an `Irregular` source.

## Control is a task, an environment, a learner and a loss

The design sketch has `paradigms/Reinforcement.py` and `training/methods/`. Here
the Battery decision problem is `environments/Battery.py`, the `Control` task keeps
the horizon's prices (the last channel) as the target so the policy never reads
them, `REINFORCE` is a learner that owns its objective (sampled actions, return
to go minus the batch mean), and evaluation is the frozen protocol with the
greedy actions scored by the `Return` loss and metric. The policy acts closed
loop on a charge grid: it emits one row of logits per charge bin for every step,
from the history alone, and the environment reads the row of the current charge;
`charge_bins=1` is the open-loop plan. The configs keep the load raw (`Identity`
scaler, the policy standardizes each window) and the price is the load floored at
zero, so returns are in load units times energy. `ForesightReturn` is the best
return with the prices known, by dynamic programming over the charges reachable
from the initial one (41 with the default battery). The `Return` losses and
metrics and the learner use `Battery` directly; a second environment needs a
selection field. `ETDatasetHour-Battery` federates the policy over ETTh1 and
ETTh2 with FedAvg's sample-weighted average; Serial and Ray give the same result.

## Generation is unconditional synthesis scored on samples

The generation tasks follow the taxonomy of the financial time-series generation
survey (Zhang et al., 2026): extrapolation is `ProbabilisticForecasting`, imputation
is `Imputation`, and synthesis is `Generation`, a DDPM (Ho et al., 2020) whose
`Denoiser` predicts the noise of a whole window. The `Generative` protocol follows
TimeDP's evaluation (TimeCraft, `metrics/metrics_sets.py`), with these differences:

- TimeDP scores univariate windows under one global z-score; KairoTS scores
  multivariate windows on each client's train-fitted scaler, and MMD flattens the
  whole window.
- TimeDP's flat KL (50 bins on the real edges, a `1e-9` floor) is replaced by the
  per-channel 1-Wasserstein distance, which stays finite when samples leave the
  real range.
- The survey's temporal and cross-sectional checks are `ACFError` (lags 1 to 10)
  and `CorrelationError`; `DCR` is its memorization check, as a ratio to the
  distance of held-out real windows.
- Real windows take TimeDP's stride (`output_length`), so each client scores about
  `test rows / output_length` windows; `DCR` searches every train window.
- The sample metrics score the test phase only; validation reports the denoising
  loss, which picks the best round.
- TimeDP clamps each step's clean-window estimate to [-1, 1] (its centered-PIT
  scaling, 200 steps, betas 5e-4 to 0.1 on a square-root grid); KairoTS clamps the
  z-scored estimate to `clip = 5` and keeps DDPM's 1000 linear steps (1e-4 to
  0.02). Without the clamp the reverse process divides by `sqrt(abar_T)` overall
  (about 158) and an undertrained denoiser's samples blow up (sample std 11 against
  1 on ETDatasetHour).

- TimeGAN's discriminative and predictive scores train a GRU classifier and a GRU
  next-step predictor. Here `DiscriminativeScore` is a leave-one-out
  1-nearest-neighbour two-sample test (`|accuracy - 0.5|`), and `TSTR` fits a
  one-step linear autoregression (4 lags) by least squares on the samples and on
  the train windows and reports the ratio of their held-out MSEs: both are closed
  form, so scoring trains no second model.

- The survey's stylized facts of returns are `KurtosisError` (excess kurtosis),
  `VolatilityClusteringError` (ACF of `|x|`), `LeverageError` (`corr(x[t],
  x[t + tau]^2)`, lags 1 to 5), `VaRError` and `ESError` (the 5% lower tail), each
  the mean absolute real-minus-generated difference per channel. They are not in
  `Generation.METRICS`; a config lists them. They read the scaled windows, so the
  scaler's shift and scale enter VaR and ES. The tail index (Hill estimator) is
  not computed: it needs a choice of the number of order statistics. At the 5%
  level a Gaussian of the real variance has a VaR and ES close to a
  heavy-tailed series', so these two mostly catch a wrong scale or skew.

`GoogleStock` is TimeGAN's `stock_data.csv` (the TSGBench Stock set). TimeGAN's
loader reverses the rows before windowing; KairoTS keeps the file's chronological
order, so the time split puts the latest days in test. With `returns`, every
channel is its daily log return. These returns are heavy tailed (excess kurtosis 9
to 21 per channel; the largest move is about 16 standard deviations), so the
stock config sets `clip = 20`.

Not computed: conditional generation. Only `Denoiser` reads the diffusion step
channel, so another model fails on its input width.

## Resume is per round

With `save_checkpoints`, every finished round writes `checkpoints/resume.pt`: the
global and kept client parameters, every client state (optimizer, scheduler,
model), the accepted operations, the rows so far, the byte count, the best
validation round and the torch random state. `runtime.resume` names the run
directory; the run reloads that state, drops the rows of the interrupted round and
continues, so its rounds equal an uninterrupted run's. A completed run deletes
`resume.pt`. Job seeds are derived from the run seed, the round and the client, so
no worker random state needs saving. The prequential regime (Online)
run inside one evaluation pass and restart their replay buffer and stream there,
so a round boundary holds no stream state. A round interrupted mid-way is redone
from its start. A resume is refused, before anything is written, when the resolved
config (apart from `runtime.resume`) or the prepared data (the manifest's raw-file
hashes, splits and scalers) differ from the run's.

## Training batches are gathered on every pass

A client update does not hold its training windows: each pass (epoch) gathers them
again, replaying the job's generator from one saved state, so every pass sees the
same shuffle, augmentation noise and imputation masks. Augmentation noise and masks
are drawn batch by batch, so with both on their draws interleave. Pooled training
(`Centralized`) shuffles whole batches across clients and holds one pass's batches.


## Class-incremental training and FLwF-2

`training.name = "ClassIncremental"` divides classification labels into consecutive,
disjoint class groups. `strategy.iterations` rounds run per group; each round
receives a disjoint share of that group's training examples. Scalers fit only
on the first group's training samples, so a later class cannot affect earlier
training. Optimizer, teacher and reservoir state belong to the client checkpoint.
`continual.json` stores a lower-triangular end-of-task accuracy matrix `R`, its
final average and forgetting (best previous accuracy minus final accuracy on old
tasks). `paper_round_average_R` additionally averages accuracy over each task's
communication rounds, as in Usmanova et al. (SmartComp 2022), section 4.1.

`FLwF2` implements that paper's classification Algorithm 1. Every student starts
from the server parameters. The frozen teachers are the current server model and
the previous communication-round client model; the first round has only the
server teacher. The objective is `alpha * class_loss + beta * client_loss +
(1 - alpha - beta) * server_loss`. Each term sums cross entropy over samples and
classes. Distillation uses temperature-scaled probabilities, with no extra
`temperature**2` multiplier. Defaults are temperature 2, alpha 0.001, beta 0.7.
Teacher parameters stay client-local and are not network messages. `personalization`
scores local models on seen classes; `client_generalization` scores them on all
classes, and `generalization` scores the server model on all classes. The
BasicMotions configs exercise the algorithm; they do not reproduce the paper's
six-class UCI HAR/CNN benchmark.

## Cross-dataset zero-shot forecasting

`ZeroShot` scores a pretrained, channel-independent DLinear once, with no client
updates or communication. Every checkpoint parameter and buffer must load, and
source and target model configs must match except for `input_channels`. Results
use `cross_generalization_test_*` keys. ExchangeRate has eight currencies and
7,588 daily rows from 1990-01-01 through 2010-10-10 in the TSLib CSV.

Run `configs/forecasting/ExchangeRate-ZeroShot.json` directly. Its
`strategy.source_config` names `configs/forecasting/ETDatasetHour-DLinear-Source.json`.
The source config must describe an independent, ungrouped training run with the
same model settings except for channel count. Source training saves checkpoints.
A completed run is reused only when its resolved config, current prepared-data
keys and complete checkpoint match; failed, partial, incompatible and damaged
runs are skipped. Otherwise the source trains first. Source config paths resolve
from the working directory, like other config paths.

An explicit saved run selected with `--initialization.run_dir` takes precedence.
The target's resolved config and manifest record the concrete source run, so a
target resume keeps it even if the source config changes or disappears. Source
runs use their own configured run directory; target run options do not change
source training. Target scoring still has no training jobs or communication.

The target scaler fits on ExchangeRate's training rows. It is target-domain
preprocessing, not target model fitting. `Pretrained` enforces its calendar cut
for same-named clients and for other clients of the same dataset family. Different
known families skip only the cross-population comparison; the manifest records
that reason and the source run. Unknown, composite and grouped families keep the
conservative calendar check.

## Gaussian likelihood and tail magnitude scoring

`LikelihoodForecasting` selects DLinear's `Gaussian` head, which returns a mean
and positive standard deviation per position. `GaussianNLL` includes the Gaussian
normalizing constant and ignores non-finite targets. The mergeable `GaussianNLL` metric
uses the same density. Inverse scoring transforms means and standard deviations
separately, so the standard deviation is never shifted by the scaler mean.

`HillError` compares the per-channel Hill extreme-value index of real and
generated magnitudes. The estimator is the mean log excess of the largest `k`
values over the next order statistic, estimating xi = 1/alpha, not alpha. Its
standalone `hill` function requires an explicit `tail_count`; the metric uses
50. At least 51 positive tail observations are needed for that metric setting.
The formula is Hill (1975), as stated in Kim and Lee (Bernoulli 2009), Eq. (2.4).
This finite-sample comparison does not test Pareto-tail assumptions or select an
optimal threshold.


## Conditional diffusion source choices

CSDI (Tashiro et al., NeurIPS 2021, arXiv:2107.03502v2) follows its conditional
architecture and concatenated temporal embeddings. Its printed forward noise
coefficient and reverse prefactor conflict with its cited DDPM equations. The
implementation uses DDPM's square-root noise coefficient, reciprocal square-root
one-step alpha, and posterior variance, corroborated by official CSDI commit
`7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`. Defaults are four 64-channel blocks,
eight attention heads, 50 quadratic diffusion steps and 100 samples. The pinned
official implementation supplies omitted Transformer internals: feed-forward width
64, GELU and dropout 0.1. Random,
historical, mixed and known-test-pattern training choices are supported. Known
patterns are conditional masks (true means available). ElapsedHours can supply
explicit timestamps; otherwise positions use a unit grid.

TimeGrad (Rasul et al., ICML 2021, arXiv:2101.12072v2) follows the paper wherever
explicit: arithmetic context means, prediction-interval noise loss, ReLU in the
architecture figure, circular padding preserving the channel dimension and
32-dimensional Transformer positional embeddings. The author implementation
`zalandoresearch/pytorch-ts` at `7860c9693d55b5c086867477cc33c89485ed0167`
supplies unspecified projection sizes, categorical embedding width, LSTM dropout
and frequency lags. Daily lags are 1, 7 and 14; hourly lags are 1, 24 and 168.
The daily example uses weekday cosine/sine features. The author implementation's
absolute-mean scaling, context-plus-prediction loss, growing-frequency diffusion
embeddings and LeakyReLU in explicitly labeled ReLU blocks are not adopted.

Diffusion examples select windows at stride 512 while retaining full network,
diffusion and sample settings. This is a small runnable scenario, not a reproduction
of benchmark splits or reported performance. Centralized training keeps optimizer
state across one-epoch updates; the existing validation selector chooses an epoch.
CSDI decays Adam's rate at 75% and 90% of the configured 200 updates. TimeGrad's
100-update cap supplies candidates for validation selection, not a claim that the
paper specifies a universal epoch count. Empirical marginal and channel-sum CRPS
are reported in data units; normalized CSDI CRPS divides by total target magnitude.


The diffusion noise objectives retain the papers' squared vector norms: sum over
predicted coordinates, then average over sampled windows/time steps. CSDI does not
normalize each batch by its number of held-out coordinates as the author code does.
TimeGrad aligns calendar covariates to the previous time step in its recurrent
state, following its recurrent-state equation; author code aligns them to the
predicted step. Both choices concern method equations and remain explicit.


## Classwise Dirichlet partitions

`Dirichlet` implements Yurochkin et al.'s classwise partition (ICML2019,
arXiv:1905.12022v1), rather than Hsu et al.'s clientwise class-mixture draw.
For each label, sampled client proportions are multiplied by an indicator that
the client has fewer than N / num_clients samples, then renormalized. This follows
`IBM/probabilistic-federated-neural-matching` at
`f44cf4281944fae46cdce1b8bc7cde3e7c44bd70`, `experiment.py:partition_data`.
The balancing step changes partition ownership relative to an unbalanced draw.

The original code uses a minimum of10 samples per client and an unbounded
rejection loop. KairoTS keeps its configurable minimum (default2), rejects
impossible total sizes immediately, and limits rejection to100 full draws.
An invalid or numerically underflowed normalization raises a diagnostic instead
of creating an incomplete partition. Randomness is isolated in `default_rng(seed)`;
legacy global-RNG seeds do not identify the same draws. A direct comparison with
the author function injects the same Generator into both functions and checks
identical ownership at the author's minimum10.


## Validation direction for shared loss and metric names

A metric with the same name as the loss supplies that result key's reported
quantity. `Return` therefore reports positive reward, while its loss is negative
reward when the metric is omitted. Checkpoint selection uses the metric's
`DIRECTION` when that metric is selected for validation; otherwise it minimizes
the loss. Resume checkpoints record the selection direction. A control checkpoint
created with the incompatible minimization rule must be rerun, because its saved
best checkpoint may already be the wrong one; it cannot resume under maximization.


## Time-segment continual forecasting (protocol of Hallak and Kem)

`training.name = "TimeSegments"` follows the scenario of arXiv 2510.21491 under this
library's rules: each Beijing station is a client. The example configs use the
12 date ranges from the [pinned author notebook](https://github.com/khaledhallakk/cf-federated-timeseries/tree/f20147be55cd189a54dc8bcfea18071fc5861562),
expressed as half-open row ranges in the hourly release starting March 1, 2013.
The base spans May 1, 2013 through May 3, 2014; the 11 seasonal tasks span
May 5, 2014 through February 2, 2017. Every task is cut 80/20 by time
(`TaskWindowRatio`), windows never cross a task boundary, and the scaler fits on
task 0's training rows only, excluding the unused prefix. May 4, 2014 is unused:
the author notebook treats it as a separate base test day, whereas these configs
retain a test share within the base task. Explicit `task_ranges` preserve the
unequal seasonal lengths; uniform 91-day approximations do not match those dates. Numbers are not compared with the paper: its released
code cannot build the task loaders, scales with all rows including test periods, and
selects hyperparameters on test sets. Deliberate differences:

- `P` is 12 by 12: the base task is task 0, not a separate initialization, and
  AF, AP and AvgPerf use the paper's formulas on this matrix.
- Every task, the base included, runs the same number of rounds (20 in the example
  configs, against the paper's 500 and 30), one local epoch per round, FedAvg.
- RMSE is the square root of the pooled test MSE of the standardized target.
- DLinear on the target channel only (TEMP, 12 hours to 6) replaces the paper's
  LSTM on all features; DLinear does not read covariates. `wd` (categorical) is dropped.
  TEMP has the fewest missing values and a strong seasonal cycle.
- Missing values: the scaler is `Standard` with `imputation = "masked"` (declared by
  the source), and windows containing a missing value are not formed.
- Replay is the library's reservoir buffer of earlier tasks' windows, `replay_size`
  windows drawn per round; the paper's k-means coreset selection is not ported. EWC,
  LwF and SI are not implemented.


## ETT benchmark borders, drop-last and patience

`CompleteWindowRatio.cut_rows` (validation start row, test start row, last used
row) reproduces the fixed borders of the ETT loaders in the DLinear and TSLib
sources: for hourly ETTh1, 12, 4 and 4 months of 30 days (rows 8640, 11520 and
14400). Validation and test windows may look back across their start row, and rows
after the last border are never used, so a 336-to-96 split has 8209, 2785 and 2785
windows. Ratio cuts cannot give these counts, because they keep the rows after row
14400. The scaler still
fits only on rows 0 to 8640.

`view.drop_last` drops the short final batch of each shuffled training epoch, as
the source loaders do; evaluation batches are never shortened. `evaluation.patience`
stops after that many rounds without a better validation score (0 never stops);
the reported test scores are those of the best round, which equals restoring the
best checkpoint. The source selects on the mean of per-batch validation losses,
KairoTS on the pooled validation loss, which differs only through the last short
batch.

`configs/forecasting/ETTh1-Informer.json` keeps the paper model (see Informer
source choices) on these borders. The paper has a three-layer distilled encoder
with a one-layer replica, two decoder layers, full cross-attention and learned
categorical timestamp tables. TSLib's port has a two-layer encoder with one
distilling layer and a one-layer decoder, ProbSparse cross-attention and a
continuous `timeF` calendar projection. No TSLib compatibility fields exist, so a
score gap to the TSLib model reflects these architecture differences as well as
random streams, and is not evidence for or against either implementation.

## TimesNet classification on EthanolConcentration

`configs/classification/EthanolConcentration-TimesNet-*.json` follow the TimesNet paper
(Wu et al., ICLR 2023): two layers, `k` 3, `d_model` 32. The paper omits `d_ff`
(32), the six Inception kernels and dropout (0.1), taken from TSLib's
`scripts/classification/TimesNet.sh` and `run.py`. That script uses three layers for
this dataset, which conflicts with the paper's two; these configs keep two.

TSLib's loader standardizes each fold with that fold's own mean and spread, so the
test fold is scaled with test statistics; KairoTS fits the scaler on train only.
Its Ethanol-only `instance_norm` is `InstanceNormalized`: the per-channel mean over
steps is removed, then each step is divided by the spread across channels, not
over time. Padding masks are all ones because the series have equal length.

The recipe is Adam at 0.001, batch 16, gradient-norm clip 4, 30 epochs, patience 10
and `EpochHalving` (TSLib `lradj="type1"`, rate `0.001 * 0.5 ** (n - 1)` after
every fifth epoch). TSLib saves a checkpoint on `accuracy >= best`, so a tie keeps
the later epoch and resets patience; a named `evaluation.selection_metric` does the
same. TSLib also seeds every run with 2021 and cannot select on validation.

`-TestSelected` is TSLib's protocol: the checkpoint maximizing TEST accuracy is
also the reported one (`evaluation.leaky_test_selection`, default off, refused when
the split has a validation fold). Its score is optimistically biased. The
`-ValidationSelected` config holds out 20% of train per class and selects on
validation accuracy; it is the clean result, and cannot be compared with the
printed number as the same protocol.

The clean protocol is: the validation fold is 20% of train, stratified by class,
drawn with split seed 0; the scaler is fit on train only (instance normalization is per series);
the reported checkpoint maximizes validation accuracy; test is read once, for the
reported score, and never selects, stops or scales anything. It makes no claim
about a published number. The archive files are the official UEA release
mirrored on Zenodo (record 11206212), pinned by sha256.

## TimesNet imputation head

`MaskedImputation` hides positions like `Imputation` and packs the model input as
`[values | observed mask | calendar]`, hidden values zeroed. TimesNet's `Imputation`
head follows TSLib's `imputation`: the mean and spread are taken over observed
positions only (hidden zeros do not count), the calendar marks pass through a
bias-free linear time-mark embedding added to the value and position embeddings
(`calendar_channels`, four hourly `timeF` features), and the projection keeps the
time axis. The forecasting and classification heads are unchanged and refuse
`calendar_channels`. `Imputation` with `TimesNet` (no mask, no marks) is not the
paper's setup. Target windows are disjoint (2785 for validation and test on ETTh1),
while TSLib slides one step (2881).

## TAFAS adapts through the prequential path

TAFAS (Kim et al., AAAI 2025, arXiv:2501.04970; original code kimanki/TAFAS) trains
two gated calibration modules around a frozen forecaster on the forecast error of
the test stream: the first `p` steps of a mini-batch's first window, which are
observed once its last window has arrived, plus the full targets of the most recent earlier
mini-batch whose horizon has passed. Its loss needs targets, so it is not an
`Adapted` method, whose `adaptation/` objectives read inputs only. The `TAFAS`
protocol belongs with `Prequential`: it scores the frozen model first (`frozen_*`),
then walks each client's test windows in time order. Every target row is read through
one function that refuses a row whose time step the stream has not yet observed.
It does not reuse `Prequential` or `Online`: those train the whole model on released
windows and score each batch as it is predicted, while TAFAS replaces the not yet
observed part of a window's prediction after its mini-batch has adapted, so a window
is scored only once its prediction is final. The protocol therefore collects each
client's test batches in a first scoring pass and scores the stream's predictions in
a second. Windows must have stride 1 and the forecast channels must equal the input
channels. Each client starts from an identity calibration (zero maps and biases, so the
gate output is the input) and the backbone is never updated.

The main-text equations govern the period, loss and prediction adjustment. The
period is `ceil(input_length / f)` for the dominant frequency `f`; the original
code supplies a fallback of 24 for the zero bin, which the paper does not define.
The loss sums the partial-target MSE and the full-target MSE of the most recent
eligible past mini-batch before one optimizer step. Calibration and optimizer state
persist through the stream. Prediction adjustment replaces every still-unobserved
portion, including the last window's entire prediction. The printed appendix uses
floor division and excludes the last window, contradicting the main text. It also
places `zero_grad` between `backward` and `step`; the implementation clears gradients
before differentiation. The author code instead performs separate full and partial
updates. These discrepancies are recorded; they do not define the main-text method.

The final mini-batch, cut short by the end of the stream, is not updated when its
partial targets have not arrived. This preserves the paper's causal observation
rule. Each additional optimization step recalculates calibration from the raw
window. Gradients are accumulated only for calibration parameters; the backbone
stays in evaluation mode and its state is restored after scoring. The author code
runs every source forecaster inside per-window mean and standard deviation
normalization (Non-stationary Transformer); the plain DLinear pilot is a valid
backbone for the model-agnostic algorithm but does not reproduce that published
baseline. Full source-model selection or a normalization wrapper is outside the
one-seed engineering pilot.

The native adapter accepts one shared model, all forecast channels and no packed
calendar or target-subset input. These restrictions are checked during config
resolution; the stream additionally rejects nonconsecutive test windows. Calibration
parameters and optimizer state are local to a client stream and are never uploaded.

## M3 source and scoring

M3 sources read the original competition actuals from the `M3C.xls` workbook and
mark each series' final published horizon as test. The yearly, quarterly, monthly
and other-frequency variants retain their official horizons and category/start
metadata. The four Monash v3 `.tsf` archives were checked against every workbook
sequence; all 3,003 agree. The Mcomp 2.8 archive differs at one observation of N2786
(position 83: Mcomp has +1200; the workbook and Monash archive have -1200), so it is
not the source used here. `SMAPE` pooled over points equals mean per-series sMAPE
within each frequency because all series in that frequency have the same horizon;
metrics across frequencies must remain separate because their horizons differ.
Four `M3*-Naive` reference configs score the official held-out folds using the
framework's `SMAPE` and `MASE`. `M3Yearly` persistence is also the paper's Naive2
baseline because yearly seasonality has period 1; it produces mean-client sMAPE
17.8799, matching the original paper's Yearly Naive2 average over horizons 1–6
(17.88, Table 13). The Quarterly, Monthly, and Other configs intentionally use
last-value persistence only; they are not seasonal Naive2 and are not presented as
paper reproductions. Their scores are reported as separate frequency results.

## M4 source and competition metrics

M4 sources use all six official frequency groups from the competition repository
at revision `6c1067e5a57161249b17289a565178dc7a3fb3ca`, plus its `M4-info.csv`
metadata. Counts and horizons are validated from that metadata: Yearly 23,000/6,
Quarterly 24,000/8, Monthly 48,000/18, Weekly 359/13, Daily 4,227/14 and Hourly
414/48. The info file's numeric `Frequency` is the seasonal period used by MASE;
`SP` names the group. Category and starting-date metadata are preserved per series.
The source notes that these starting dates were added after the competition and
does not expose them as model inputs.

The framework's mergeable `SMAPE` and `MASE` scores cover the corresponding M4
metrics. The official `Benchmarks and Evaluation.R` at the pinned repository
revision defines OWA as the arithmetic mean of aggregate MASE and sMAPE ratios to
the corresponding Naive2 errors. Within one frequency, the horizon is fixed, so the
mean over forecast points equals the mean of each series' horizon-averaged score.
Across frequencies, the official overall result weights each series equally (the
six fold means are weighted by their series counts), not each forecast point.
Naive2 first runs the official seasonality test (seasonal period `m`, at least `3m`
observations, 1.645 ACF threshold); seasonal series are multiplicatively
decomposed, forecast by naive on the adjusted series, then reseasonalized.
Nonseasonal series use the naive forecast. The reference script requires R
`forecast` 8.2.

The pinned repository also releases `Point Forecasts/submission-Naive2.rar` and
`Evaluation and Ranks.xlsx`. The extracted forecast archive has all 100,000 unique
IDs and the expected per-frequency horizons. Against the pinned train/test files,
its sMAPE and MASE match the official workbook at all six frequencies and overall
at the workbook's printed precision. The workbook's Weekly Naive2 sMAPE is 9.161;
a later N-BEATS supplement prints 9.191 for that row, so the competition's pinned
workbook is the score authority here.

KairoTS OWA implementation must score each frequency against its matching Naive2
reference and aggregate the six per-frequency means by series count for the overall
competition result. A single-fold OWA must not be presented as the overall score.

## M5 source and WRMSSE

`M5Validation` and `M5Evaluation` use the official competition's 30,490
product-store series, 28-day actual holdouts and fold-specific weights. The raw
archive also retains the calendar and selling-price files. It is pinned to the
Mcompetitions-linked shared dataset; the official `weights_validation.csv` and
`weights_evaluation.csv` define the per-series dollar-sales weights, avoiding an
independently reconstructed weighting period.

`M5WRMSSE` computes RMSSE against training differences after the first positive
sale, as in the M5 evaluation R script. It aggregates forecasts from the bottom
product-store level to all 12 official levels, applies each level's released series
weights, then equally averages the 12 level scores. This is the point-forecast
WRMSSE, not the uncertainty track's WSPL. The source presents the bottom-level
series as one multivariate client so the complete hierarchy can be scored in one
pass. The paper is Makridakis et al., “M5 accuracy competition: Results, findings,
and conclusions,” *International Journal of Forecasting* 38(4), 1346–1364 (2022),
DOI 10.1016/j.ijforecast.2021.11.013; the Competitors' Guide and competition's
`Estimate WRMSSE.R` provide the scoring and file-level details.
The `M5Evaluation-Naive` reference config repeats each product-store's last
training observation for the next 28 days, then aggregates bottom-up. Its official
evaluation fold WRMSSE is 1.7520099, matching the paper's Appendix B Naive average
of 1.752 to the reported precision. The validation actuals give 1.4638557 and are
kept as a separate public-fold score, not substituted for the final result.
