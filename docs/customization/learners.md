# Learners

A learner runs local optimization on one client: optimizer, precision and epoch loop.
Strategies call it through `Client.fit`; a strategy's extra objective (a proximal term) comes in as `regularizer`.

**Copy:** `src/kairots/learners/Gradient.py`

## Contract

| Member | Purpose |
|---|---|
| `Config` | `epochs`, `learning_rate`, `optimizer`, `precision`, ... |
| `optimizer(*, model) -> Optimizer` | build the optimizer |
| `fit_epoch(*, model, optimizer, batches, device, loss_function, regularizer) -> float` | one pass over prepared batches; mean train loss |

`Stateless` returns no optimizer and has zero gradient epochs.
Parameter-free models such as `Naive` use it; `FedRidge` also selects it because the strategy computes sufficient statistics directly instead of running gradient updates.

`Estimator` fits models with their own training procedure; copy `Estimator.py`, select with `"learner": {"name": "Estimator"}` and `"model": {"name": "XGBoost"}`, and use `LocalOnly` or `Centralized` only.

## Select it

```json
"learner": {"name": "Gradient", "epochs": 1, "learning_rate": 0.0001}
```

```json
"model": {"name": "Naive"}, "learner": {"name": "Stateless"}
```

## Schedulers

A scheduler (`src/kairots/schedulers/`, top-level `scheduler` namespace) builds a torch `LRScheduler` with `build(*, optimizer, iterations)`.
By default the worker steps it once after each client update and keeps its state in `ClientState.scheduler_state`, private and unbilled like the optimizer state; a Config field `span = "round"` makes it restart every client update and step after each optimizer step, with `iterations` the update's epochs times batches.
`Constant` is the default; `Cosine` anneals to `minimum_rate` over `strategy.iterations`.

```json
"scheduler": {"name": "Cosine", "minimum_rate": 0.0}
```


`Denoising` uses the same optimizer contract as Gradient and calls the conditional model's `training_loss` instead of sampling during training.
CSDI and TimeGrad require this learner in fp32.
Historical CSDI patterns are initialized from training batches before the pass.
`MultiStep` decays the rate by `factor` at the configured fractions of the total update count.

`EpochHalving` is TSLib's `type1` rule: after every `interval`-th client update `n` the rate is `learning_rate * 0.5 ** (n - 1)`.
`Gradient.max_gradient_norm` (0 = off) rescales each update's gradient to that 2-norm.
