# Augmentations

An augmentation changes training inputs only (`batch_x`), never targets, validation or test data.
The worker applies it to every training batch with the job's seeded generator, so each round draws fresh noise and Serial and Ray stay identical.
It is off unless `data.augmentation` is set.

**Copy:** `src/kairots/augmentations/Jitter.py`

## Contract

| Member | Purpose |
|---|---|
| `Config` | nested dataclass, `name` first |
| `apply(*, batch_x, generator) -> Tensor` | the augmented inputs, same shape |

## Select it

```json
"data": {"augmentation": {"name": "Jitter", "sigma": 0.03}}
```
