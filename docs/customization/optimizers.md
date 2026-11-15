# Optimizers

An optimizer component builds a PyTorch optimizer from parameters, a learning rate
and the learner's fused setting. Each selectable optimizer has a nested `Config`.

**Copy:** `src/kairots/optimizers/SGD.py`.

```json
"learner": {
  "name": "Gradient",
  "learning_rate": 0.01,
  "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 0.0005}
}
```

`Adam` is the default, with zero weight decay. `SGD` defaults to zero momentum and
weight decay. `Adapted` has its own `evaluation.optimizer` with the same contract;
its learning rate belongs to the evaluation config. Plain optimizer-name strings
are rejected. CLI overrides use `--learner.optimizer.name` and nested option paths.
