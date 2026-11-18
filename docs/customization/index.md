# Customization

KairoTS finds components by file name.
A new file in the right directory is selectable by name at once; there is no list to update.
Each page below covers one kind of component.

```{toctree}
:maxdepth: 1

metrics
models
tasks
initializers
strategies
datasets
partitions
splits
scalers
augmentations
topologies
learners
optimizers
training
```

## See what you can choose

The [component catalog](../reference/catalog.md) lists every option.
The same list comes from the registry, which reads files without importing them:

```bash
python -m kairots list                         # every category and component
python -m kairots list metrics --task Forecasting
```

```python
from kairots.registry import info, names

names(category="metrics", task="Forecasting")  # ("MAE", "R2")
info(category="metrics", name="R2")            # {"TASKS": ..., "PHASES": ("test",), ...}
```

## Where a component can be used

A component says where it applies with plain class constants.
The registry reads them with `ast`, and `Experiment` rejects an incompatible choice before any data is prepared.

| Constant    | Declared by | Meaning |
|-------------|-------------|---------|
| `ROLES`     | metrics (required) | `loss`, `metric`, or both; checked where the component is selected |
| `TASKS`     | metrics (required); any component (optional) | tasks it supports; absent means every task |
| `NEEDS`     | metrics (required) | what it reads: `point_prediction`, `target`, `mask`, `distribution`, `labels`, `scores` |
| `REQUIRES`  | components (optional) | dotted paths to allowed tuples; singleton requirements fill omitted fields, explicit conflicts are rejected |
| `DIRECTION` | metrics (required) | `min` or `max`; used by monitoring and best checkpoints |
| `PHASES`    | metrics (optional) | phases it may be selected for: `train`, `validation`, `test`; absent means all |
| `ABSTRACT`  | families, bases | `True` hides the class from selection |
| `LOSS`, `METRICS` | tasks (required) | default `evaluation.loss` and per-phase metrics |

For example, `REQUIRES = {"learner.optimizer.name": ("SGD",)}` selects SGD when the file and CLI omit the optimizer name.
An explicit Adam is rejected, even though Adam is the framework default.
Explicit compatible values are preserved.
Several allowed values only restrict a field; they never choose it.
Required component names resolve before their options and can introduce more requirements.
Conflicting requirements fail instead of depending on selection order.
Client groups resolve omitted settings independently from their original configuration.
Missing optional namespaces read as `None`; lists compare as tuples.
Use `validate(*, experiment)` only for relations such as a period count bounded by the dataset's number of classes.
`GROUPS = True` declares evaluation support for client groups.

These checks also cover nested selections, including optimizers, replay, topology, online regimes and adaptation methods.
A component in a nested config has the same `TASKS`, `REQUIRES` and `validate` contract as a top-level selection.

The config-contract tests audit all registered requirement declarations, even when their component is absent from example configs.
They start with default settings and select the owning Config and its singleton name requirements before checking conditional fields, such as SGD momentum.
Paths must exist, allowed values must match their field types, and `None` requires an optional field or namespace.

Shared settings inherit a narrow Config base rather than introducing another home for the same knob.
Profile overrides keep their method's defaults.

| Settings | Shared home |
|---|---|
| Diffusion levels and beta endpoints | `models/base.py::Diffusion.Config` |
| Conditional trajectory sample count | `models/base.py::ConditionalDiffusion.Config` |
| Replay buffer capacity and host draw options | `replay/base.py::Buffer.Config` and `Replay.Config` |
| Online evaluation options | `training/Online.py::Online.Config`, reused by `Prequential` |
| Gradient learning rate and optimizer choice | `learners/base.py::Learning.Config` |
| Learner pass count | `learners/base.py::Learner.Config` |
| Optimizer weight decay | `optimizers/base.py::Optimizer.Config` |
| Online/adaptation step budget | `optimizers/base.py::UpdateBudget.Config` |
| Executor device, workers, determinism and persistence | `runtime/executors/base.py::Executor.Config` |
| Evaluation loss and metric selectors | `evaluation/base.py::Scoring.Config` |
| Strategy update count | `strategies/base.py::Strategy.Config` |
| Proximal penalty coefficient | `strategies/Proximal.py::Proximal.Config` |
| Calendar selection | `tasks/base.py::Timed.Config` |
| Partition counts, class coverage and concentration | `data/partitions/_labeled.py` |
| Window lengths, minimum counts and stride | `data/splits/base.py` |
| Single-source transform selection | `data/sources/base.py::SingleMember.Config` |
| Network widths, residual depth, dropout and embedding dimensions | `models/base.py` |

The ownership test scans all Config declarations.
Unrelated paper coefficients and random streams have exact reviewed owners; model dimensions derived by the schema have a model-category exception.
A new component cannot acquire an unrelated exception merely by reusing its field name.
Private adapter Configs receive the public recipe's values and have explicitly reviewed declaration pairs.

Generation uses `task.diffusion_steps`, matching the conditional models' field.
XGBoost uses `model.shrinkage` for its boosting shrinkage; `learning_rate` is the gradient optimizer's step size.
These replace Generation's former `steps` and XGBoost's former `learning_rate` keys.
XGBoost declares the native `Estimator` learner as a requirement: omission selects it automatically, while an explicitly conflicting gradient learner is rejected.

## Rules for every component

1. The class name equals the file name (`metrics/MAE.py` defines `MAE`); a model's name is
its directory (`models/DLinear/`).
2. Settings live in a nested `Config` dataclass with `name` as its first field.
3. Every parameter is keyword-only and type-hinted (`def update(self, *, prediction: Tensor)`).
4. Copy the reference file named on the component's page; reference files are real,
tested components.
5. Follow the [conventions](../reference/conventions.md): names, glossary, words.
6. Run `python scripts/format.py`, `python -m pytest` and `python scripts/docs.py`.
