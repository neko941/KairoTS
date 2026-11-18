# Tasks

A task defines what an example means: it prepares the model input and target and declares the model head, default loss, metrics and evaluation protocol.
Tasks are selected with the top-level `task.name` field.

**Copy:** `src/kairots/tasks/Forecasting.py` for a task with covariates, or `src/kairots/tasks/Classification.py` for a simple task.

## Contract

| Member | Purpose |
|---|---|
| `Config` | Nested keyword-only dataclass; put `name` first and validate task-specific settings in `__post_init__`. |
| `HEAD` | Model head name required by the task, such as `Forecasting` or `Classification`. |
| `RECONSTRUCTS` | Whether the task trains on a target window reconstructed from the input. |
| `LOSS` | Default loss component name. |
| `METRICS` | Default metric component names. |
| `PROTOCOL` | Optional evaluation protocol for tasks that cannot use the default frozen protocol. |
| `prepare(*, batch_x, batch_y, output_length, generator=None)` | Return `(model_input, target)` while preserving batch alignment and preventing unavailable future values or hidden labels from entering the model input. |

The schema fills shape fields derived from the data and task.
Keep a channel's meaning in the task config, and use `REQUIRES` or `validate(*, experiment)` for compatibility rules that cannot be expressed by a field alone.
Timed tasks can inherit `Timed.Config` to expose calendar feature names.

## Add a task

1. Add `src/kairots/tasks/<TaskName>.py` and define a class with the same name.
The lazy registry discovers it from the file name; do not add an import list.
2. Declare `HEAD`, `RECONSTRUCTS`, `LOSS` and `METRICS`.
Declare `PROTOCOL` only when the task requires a non-default protocol.
3. Implement `Config` and `prepare`.
Use the provided generator for any random masking or sampling; do not read validation/test labels to prepare training examples.
4. Add tests for prepared shapes, target alignment, and the task's information
boundary.
Add a short end-to-end config when the task/model combination is supported.
5. Build the docs and run the relevant integration and leakage tests in the project
test environment.

For example, `Forecasting.prepare` appends known-future calendar features while masking future targets and past-only covariates.
`MaskedImputation.prepare` appends the observed-value mask and calendar channels.
Reuse those patterns where the data contract matches; do not create a task variant just to rename a model head.

## Select it

```json
"task": {
  "name": "Forecasting",
  "target_channels": ["OT"],
  "calendar": ["MonthOfYear", "DayOfMonth", "DayOfWeek", "HourOfDay"]
}
```

See [Models](models.md), [Losses and metrics](metrics.md), and [Training](training.md) for the components that must agree with the task.
