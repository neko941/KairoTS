# Losses and metrics

The `metrics/` category holds both objectives and reported scores. Each component
declares `ROLES = ("loss",)`, `("metric",)` or `("loss", "metric")`. Configuration
rejects a component used in a role it does not support.

A loss implements `__call__(*, prediction, target) -> Tensor`, returning a batch
mean. The learner trains with `evaluation.loss`; validation `pooled_<loss>` selects
the best checkpoint, so lower must be better. Calling a loss does not accumulate
metric state. `Return` is a scoring-only loss because greedy actions have no gradient.

**Loss example:** `src/kairots/metrics/MSE.py`.

A metric scores predictions during evaluation. It must be **mergeable**: per-client
accumulators merge into the exact pooled value, so pooled and mean-client numbers come
from one pass over the data.

**Copy:** `src/kairots/metrics/MAE.py`; `metrics/R2.py` shows a phase limit.

`MASE` scores absolute error against the history's mean seasonal difference.

`M5WRMSSE` is the M5 Accuracy competition score. It compares one 28-day forecast
against the official per-series RMSSE scales and weights, sums the product-store
forecasts to all 12 hierarchy levels, computes each level's weighted score, and
averages those 12 scores equally. It requires the complete M5 bottom-level series
as one client; use it with `M5Validation` or `M5Evaluation` and `PerFile`.
`configs/forecasting/M5Evaluation-Naive.json` is the parameter-free last-value
benchmark: its final-fold score is 1.7520099, matching the M5 paper's Appendix B
Naive WRMSSE of 1.752 to the table's printed precision.

`M4OWA` reports the M4 sMAPE/MASE average relative to the pinned Naive2 baseline
for one frequency fold. It is test-only. The overall competition OWA combines the
six fold means weighted by their series counts; a single-fold value is not the
overall score. `M4OWA` checks that the configured source seasonality and horizon
match the corresponding M4 reference. `overall_owa` combines six pairs of per-fold
`pooled_smape`/`pooled_mase` values into the overall score.

## Contract

| Member | Purpose |
|---|---|
| `ROLES`, `TASKS`, `NEEDS`, `DIRECTION` | required for a metric role (see [Customization](index.md#where-a-component-can-be-used)) |
| `PHASES` | optional; `("test",)` makes a report-only metric |
| `Config` | nested dataclass, `name` first |
| `update(*, prediction, target)` | add one batch to the accumulators |
| `merge(*, other)` | add another instance's accumulators |
| `compute() -> float` | final value; raise `ValueError` with no observations |

Accumulate sums, not means, and keep them as tensors on the batch device; `compute` is
the only host sync.

## Select it

Metrics are chosen per phase. `train` scores the frozen global model on train windows;
validation runs when the split has validation windows, and then picks the best
checkpoint.

```json
"evaluation": {"loss": "MSE", "metrics": {"validation": ["MAE"], "test": ["MAE", "R2"]}}
```

A metric whose `PHASES` exclude a phase fails before the run starts:
`metrics 'R2' cannot be selected for the validation phase; it allows ('test',)`.

## Test it

Check that two merged halves equal the pooled value, and compare against a direct
formula; see `test_r2_merges_exactly_and_is_test_only` in `tests/unit/test_metrics.py`.
