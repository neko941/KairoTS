# Training schedules

Training schedules choose which training windows an update can use. They share the
worker's batching, learner, optimizer and strategy-specific fitting code.

| Schedule | Training data | Persistent state |
|---|---|---|
| `Offline` | Full training split on every update, for any task | None |
| `ClassIncremental` | Disjoint round releases from consecutive classification-label groups | Optional replay in client state |

| `TimeSegments` | Forecasting tasks as consecutive time segments; each round trains on the whole current task | Optional replay in client state |

`Continual` in `training/base.py` is the non-selectable base for period schedules.

Omitting top-level `training` selects ordinary offline behavior. It can also be
stated explicitly:

```json
"training": {"name": "Offline"}
```

Class-incremental training requires classification, two or more periods, and the
`FedAvg` or `FLwF2` strategy. Rounds run per period. Its scaler sees only the first
round's released examples and stays fixed; replay never admits future examples.

```json
"training": {"name": "ClassIncremental", "periods": 2, "replay_size": 4}
```

`TimeSegments` pairs with `data.split.name = "TaskWindowRatio"`; `base_length` and
`task_length` and `task_ranges` must match in both. With no explicit ranges,
the split cuts every full segment and the schedule releases the first `periods`.
With `task_ranges`, each half-open row pair defines one period; their count must
equal `periods`. Rows outside those ranges are unused. Task 0 is the base segment and every task is cut
80/20 chronologically. The scaler fits on task 0's training rows only. `continual.json`
holds the RMSE matrix `P` (`P[i][j]`: model after task `i` on task `j`'s test windows) with
`AF`, `AP` and `AvgPerf` (Hallak and Kem, arXiv 2510.21491). Naive training is
`replay_size: 0`.

```json
"training": {"name": "TimeSegments", "periods": 12, "base_length": 8760, "task_length": 2184, "replay_size": 512}
```

`Online` is selected by `Prequential` evaluation, where predictions precede updates
and targets must have arrived before training. It is separate from the offline
training schedule and can be combined with implicit or explicit `Offline` training.

## Component contract

Copy `training/Offline.py`. Its `select` method receives the client's training
windows, round index, rounds per period, optional class count, client state and
seed. It returns the windows available to the update. After fitting, `commit`
updates any schedule state. The worker owns the common fitting path.

`Continual` owns period releases, replay configuration and replay state. A concrete
subclass supplies `windows` and declares `CONTINUAL = True` literally, so both the
registry and runtime use its period scoring and first-release scaler boundary.
`ClassIncremental` defines consecutive label groups and classification-specific
`TASKS`, `REQUIRES` and `validate`; these constants are never inherited by registry
discovery. Keep applicability rules on each selectable component.
