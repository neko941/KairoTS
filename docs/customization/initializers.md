# Initialization policies

An initialization policy adjusts the canonical model once after construction and
before clients receive it. The run manifest records the policy, seed and tensor
names it changed. Client workers load their assigned state; they do not apply the
policy again.

**Copy:** `src/kairots/initializers/Rules.py` for pattern-based initialization, or
`src/kairots/initializers/Pretrained.py` for loading a saved checkpoint.

## Contract

| Member | Purpose |
|---|---|
| `Config` | Subclass `Policy.Config`; keep `name` first and validate options in `__post_init__`. |
| `MODELS` | Optional tuple restricting the model recipes that can select this policy. |
| `apply(*, model, manifest=None) -> dict[str, str]` | Apply the policy in place and return a mapping from each changed state-tensor name to its source or initializer name. |

The base policy applies ordered `RULES` to `model.named_parameters()`. A rule that
matches no parameter or overlaps another rule is an error; unmatched parameters
keep the model recipe's native values. Tied parameters appear once in
`named_parameters()` and are initialized once. Policies that load buffers as well
as parameters should report every loaded state-tensor name.

## Add a policy

1. Add `src/kairots/initializers/<PolicyName>.py` with a class of the same name.
   The registry discovers the new file; no central list needs editing.
2. Inherit `Policy`, define a nested Config and override `apply` only when the
   standard ordered-rule behavior is not enough. If the policy is model-specific,
   declare `MODELS` so incompatible configs fail during schema validation.
3. Return exactly the tensor names changed. Keep initialization on the canonical
   model in setup; do not initialize in a client worker, evaluation protocol or
   model forward call.
4. Test matched/unmatched names, overlap behavior, tied parameters, and preservation
   of native values outside the policy's scope. If loading a checkpoint, test that
   new or shape-incompatible heads retain their native initialization and that
   held-out data cannot enter pretraining.
5. Run the initializer, checkpoint and run-artifact tests; build docs and run the
   focused integration tests in the project test environment.

`Rules` accepts parameter patterns, an initializer name from
`initializers.base.INITIALIZERS`, and its options. For a DLinear classifier with
average body weights and a Xavier head:

```json
"initialization": {
  "name": "Rules",
  "rules": [
    ["body.linear_seasonal.*.weight", "inverse_fan_in", {}],
    ["body.linear_trend.*.weight", "inverse_fan_in", {}],
    ["head.projection.weight", "xavier_uniform", {}],
    ["head.projection.bias", "zeros", {}]
  ]
}
```

`Average` selects linear weights, sets them to `1 / fan_in`, and preserves biases
and other layers. `Pretrained` loads matching names and shapes from a saved
checkpoint; unmatched parameters keep their constructor values. Its manifest
supplies the held-out timestamp check, so a source run cannot train past the
target's first protected period.

See [Models](models.md) for recipe-native initialization.
