# Python API

Infrastructure modules. Each selectable component has its own page in the
[component catalog](catalog.md).

```{eval-rst}
.. automodule:: kairots.registry
   :members:

.. automodule:: kairots.config.schema
   :members: Experiment

.. automodule:: kairots.runtime.orchestrator
   :members: fit, evaluate, run
```

## Separate source fitting and target evaluation

```python
import json
from pathlib import Path

from kairots import Experiment, evaluate, fit

values = json.loads(Path("configs/forecasting/ETDatasetHour.json").read_text())
source = Experiment.from_dict(values=values)
fitted = fit(experiment=source)

target = Experiment.from_dict(values=values)
result = evaluate(experiment=target, source_run=fitted.run_dir)
```

`fit()` exposes source `train` windows and validation `support` for checkpoint
selection; it withholds test windows and later series rows from the worker data.
`evaluate()` accepts only a completed `fit()` artifact, loads its validation-selected
checkpoint (or final checkpoint when no validation selection exists), performs no
target training, and scores only target `test` query windows. Frozen prediction and
TTA updates receive query inputs; query labels go to metrics after prediction, or
are released after scoring in the prequential protocol. The run manifest records
these permissions. Use `run()` for the existing combined training-and-evaluation
path; an explicitly leaky historical selection option remains an opt-in of that API.
