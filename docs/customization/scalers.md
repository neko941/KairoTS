# Scalers

A scaler normalizes each client's values. It is fitted on that client's **train period
only**, so validation and test never leak into it.

**Copy:** `src/kairots/scalers/Standard.py`

## Contract

| Member | Purpose |
|---|---|
| `Config` | nested dataclass, `name` first |
| `fit(*, values, config) -> Scaler` | classmethod; per-channel statistics from train rows |
| `validate_values(*, values, config, source, client, channels)` | optional; validates all client rows before split filtering |
| `transform(*, values)`, `inverse_transform(*, values)` | apply and undo |
| `mean`, `scale` | fitted statistics, stored in the prepared split summary |

## Select it

```json
"data": {"scaler": {"name": "Standard"}}
```

Standard rejects null or NaN values by default, including validation and test rows.
The error identifies the source, client and channel. To retain missing observations
for a model that reads masks, select `"imputation": "masked"`: statistics skip
missing entries, an unobserved channel keeps mean 0 and scale 1, and values stay
NaN. This does not fill values. Irregular sources require this mode; omitted values
are filled by their requirement, while an explicit conflicting value is rejected.
