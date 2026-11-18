# KairoTS development map

Read these before changing code; this file only points to them.

| Need | Where |
|---|---|
| Package layout, data shapes, runtime contracts | `docs/explanation/architecture.md` |
| Names, glossary, code rules, words | `docs/reference/conventions.md` (checked by `scripts/check_style.py`) |
| Component contracts and reference files to copy | `docs/customization/<kind>.md` |
| Rules, gate commands | `docs/contributing.md` |
| Why things are the way they are | `docs/explanation/notes.md` (describe each decision without dated headings or process chatter) |
| Traps met before, and what to do instead | `docs/reference/pitfalls.md` |

## Commands

Use the repo env `.conda` (`.\.conda\python.exe` on Windows).

```powershell
.\.conda\python.exe scripts/format.py      # ruff fix + format + style check (--check: report only)
.\.conda\python.exe -m pyright
.\.conda\python.exe -m pytest -q
.\.conda\python.exe scripts/docs.py        # warnings are errors
.\.conda\python.exe scripts/smoke.py       # every config, one iteration
.\.conda\python.exe scripts/leakage.py     # training is identical with a poisoned test set
.\.conda\python.exe -m kairots list [category] [--task Forecasting]
.\.conda\python.exe -m kairots report [ROOT] [--metrics a,b]
```

## Workspace

- Git is local only (no remote); do not commit unless asked.
- License: MIT-0 (`LICENSE`); new files need no header.
- `datasets/`, `runs/` and `tmp/` are untracked. `tmp/Time-Series-Library` is a reference
  checkout of `thuml/Time-Series-Library` pinned at
  `4e938a1767106324dd753b2a44832bf870a0252e`.

## Additional composition

- Settings and local bindings use full words: window `*_length`, feature
  `*_dimension`, `moving_average_kernel`, `encoder_layers`, `num_inception_kernels`.
  TimesNet's `k` and Vaswani's `d_model`/`d_ff` retain their cited paper symbols.

- Model parts stay in their model's directory; blocks used by two or more models live in `src/kairots/layers/`, except `iTransformer/InvertedTransformer.py`, the shared backbone inherited by the four inverted-attention variants.
- Shared Config homes live in narrow category `base.py` modules (and the partition
  helper); see the settings-home table in `docs/customization/index.md`. Profiles
  inherit them and override defaults. `Generation` uses
  `diffusion_steps`; XGBoost uses `shrinkage` for its boosting coefficient and
  declares the native `Estimator` learner through `REQUIRES`.
- Singleton `REQUIRES` values fill fields omitted by the config and CLI. Explicit
  conflicts fail, including values equal to framework defaults; multiple allowed
  values never select a default. Groups retain their own omission tracking.
- `Standard` rejects missing values across every split unless
  `data.scaler.imputation` is `"masked"`. Irregular sources require that mode;
  it preserves missing values for models that read masks.
- Ordinary training uses `Offline` implicitly or through `training: {"name": "Offline"}`.
  `ClassIncremental` owns disjoint classification releases and checkpointed replay;
  `Continual` is the non-selectable base for period schedules. Rounds run per period. Online stream
  replay belongs to `Prequential` evaluation (`regime: Online`, `replay_size`, default 0).
- `ZeroShot` scores a complete channel-independent DLinear checkpoint on another
  dataset, with `cross_*` metrics and no target jobs. `strategy.source_config` reuses
  a completed matching source run or trains it first. An explicit
  `initialization.run_dir` takes precedence. The selected run is pinned in the
  target's resolved config; source runs use their own config's `runtime.runs_dir`.
- `LikelihoodForecasting` uses DLinear's Gaussian head and `GaussianNLL` for both
  loss and metric roles. Objectives and scores share the `metrics/` category.
- Optimizers are registry components. `learner.optimizer` is an object such as
  `{"name": "SGD", "momentum": 0.9}`; `Adapted` uses `evaluation.optimizer`.
  `HillError` compares Hill xi of absolute generated and real values (50 tails).

- `SampleForecasting` retains `[batch, steps, channels, samples]` trajectories.
  `TimeGrad` reads lagged targets and future calendar features; `Denoising` fits
  its noise objective. `ProbabilisticImputation` encodes `[values | mask | time]`
  for `CSDI`; time is optional (`calendar: [ElapsedHours]`).
- `Informer` reads packed observed history plus calendar, builds a decoder from
  `label_length` history rows and zero future values, and retains future marks.
  Calendar order is MonthOfYear, DayOfMonth, DayOfWeek, HourOfDay, optionally
  MinuteOfHour. Its replicated encoder follows the original paper; source
  discrepancies are recorded in `docs/explanation/notes.md`.
- Diffusion example configs retain the paper networks, schedules and sample counts;
  their stride 512 selects spaced windows for runnable examples. They are not
  benchmark reproductions. Centralized updates correspond to epochs and validation
  selects the best update. CSDI uses `MultiStep` decay at 75% and 90%.

- `analysis/landscape/` separates `one_d.py`, `two_d.py`, and `three_d.py`;
  `common.py` owns shared loading, sampling and Hessian calculations. The `analysis`
  extra provides matplotlib. Two directions produce both contour and 3D views;
  3D uses loss as height. Analysis needs saved checkpoints, config and prepared data,
  without extra training logs. Training-path projection would need intermediate
  checkpoints and is not currently implemented.
