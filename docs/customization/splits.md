# Splits

A split policy turns each client's data into train, validation and test windows:
`(start, target)` positions for series, local sample indexes for panel data.

**Copy:** `src/kairots/data/splits/CompleteWindowRatio.py` (series) or `Predefined.py`
(panel)

| Split | Data | Rule |
|---|---|---|
| `CompleteWindowRatio` | series | target periods cut by `ratios`; windows never cross a timestamp gap |
| `TaskWindowRatio` | series | `CompleteWindowRatio` inside ordered time segments (`task_ranges`, or `base_length` then `task_length` rows); windows never cross a segment boundary |
| `Predefined` | panel | keeps the archive's train/test folds; draws `validation_ratio` of each class from train (floored, seeded) |
| `TestFold` | series | tiles official test and held-out validation folds; trains on earlier windows |

`Predefined.split(*, samples)` gets one row per sample with `fold` and `label`.

## Contract

| Member | Purpose |
|---|---|
| `Config` | `input_length`, `output_length`, `offset_length`, `ratios`, `cut`, `min_windows`, ... |
| `split(*, frame, ...) -> dict[str, list[tuple[int, int]]]` | windows per split |
| `shared_cut_dates(...)` | optional; one calendar cut for all clients (`cut="shared_calendar"`) |

Target periods of different splits must not overlap. A client with fewer than
`min_windows` in a split is excluded, and the reason is recorded in the run manifest.

## Select it

```json
"data": {"split": {"name": "CompleteWindowRatio", "ratios": [0.7, 0.1, 0.2]}}
```

The default is train/test 80/20 with no validation (`ratios` 0.8, 0, 0.2;
`TestFold.validation_windows` and `Predefined.validation_ratio` 0), and the final
round is reported. A validation share, as above, turns on the best-checkpoint pick.
Anomaly configs set `validation_ratio`, because their threshold is calibrated on
validation scores.

`TaskWindowRatio.task_ranges` accepts ordered, nonoverlapping half-open row pairs
`[first, last]`. When supplied, these override the uniform segment lengths and
exclude rows outside the ranges. An explicit range beyond the data is an error.
Use identical ranges in `training.name = "TimeSegments"`, one per period.
