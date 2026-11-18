# Models

A model is a **directory**: its `__init__.py` holds the recipe class named after the directory, and the other files are its parts (body, heads).
Only the recipe is selectable.

**Copy:** `src/kairots/models/DLinear/`

`TimesNet` is a second example, with a body shared by forecasting, reconstruction and classification heads.
A head owns everything specific to its task (TimesNet's forecasting normalization and time extension, DLinear's body output length); the recipe has no task branches.

`Informer` builds encoder values/marks and decoder values/marks from the packed forecasting input.
Set `task.calendar` to `MonthOfYear`, `DayOfMonth`, `DayOfWeek`, `HourOfDay`, in that order; minute data can append `MinuteOfHour`.
Its `label_length` selects the decoder's observed history.
Future values become zero placeholders; future calendar marks are retained.
The model predicts every requested target channel jointly and requires `0 < label_length < input_length`.

`configs/forecasting/ETDatasetHour-Informer.json` runs the ETTh1 member alone, 96-to-96, using the paper's network widths and eight fixed training epochs.
Its 60/20/20 chronological split and validation selection are a native sanity protocol, without paper early stopping.
It is not a reproduction of the paper's table or an interchangeable TSLib baseline.

```text
models/DLinear/
  __init__.py        DLinear: Config + wiring, picks a head by name
  Body.py            body (shared by every head)
  SeriesDecomposition.py  moving-average trend and seasonal parts, used by Body
  Forecasting.py     head
  Classification.py  head
  Quantile.py        head (ProbabilisticForecasting)
  Gaussian.py        head (LikelihoodForecasting)
```

## Shared blocks

A model imports nothing from another model's directory (checked as M1), except inverted-attention variants importing `iTransformer/InvertedTransformer.py`.
A block used by two or more models lives in `src/kairots/layers/`, one class per file named after it, with its paper cited in the docstring.
Until a second model needs it, a block stays in its own model's directory.
`layers/PositionalEmbedding.py` serves `TimesNet` and `Informer`.

## Contract

| Member | Purpose |
|---|---|
| `Config` | `name`, `head`, and the shape fields `input_length`, `output_length`, `input_channels` |
| `__init__(*, config)` | build the body and the head named by `config.head` |
| `forward(*, batch_x) -> Tensor` | `batch_x` is `[batch, input_length, channels]` |

`input_length`, `output_length` and `input_channels` are filled from the data config when the model config leaves them out.
Use the glossary names; a port renames its source's variables (`seq_len` to `input_length`, `enc_in` to `input_channels`).

## Select it

```json
"model": {"name": "DLinear", "moving_average_kernel": 25}
```

## Test it

Compare a forward pass against the reference implementation's math on a fixed input; see the DLinear tests in `tests/unit/test_models.py`.

## Initialization policies

A recipe constructor provides native starting values.
An optional policy changes them once on the canonical model before training.
See [Initialization policies](initializers.md) for the policy contract and steps to add one.
