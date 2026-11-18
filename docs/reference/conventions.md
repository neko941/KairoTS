# Conventions

The rules below are checked by `scripts/check_style.py` (rule ids in brackets) and ruff, so a mistake fails `python scripts/format.py` instead of reaching review.
Add a rule here and to the checker together.

## Files and directories

| Kind | Name | Example |
|---|---|---|
| Component (selectable) | PascalCase file, defines the class of the same name [N2] | `metrics/MAE.py` defines `MAE` |
| Family (abstract) | PascalCase, same rule, plus `ABSTRACT = True` | `data/sources/Multivariate.py`, `data/sources/Composite.py` |
| Dataset | PascalCase directory; `__init__.py` defines the dataset or family class, variants have their own files | `data/sources/ETDataset/`, `data/sources/Electricity/` |
| Model | PascalCase directory; `__init__.py` defines only the recipe class; every other class has its own file named after it | `models/DLinear/Body.py` |
| Infrastructure | lowercase module; never defines a class named after itself [N2] | `base.py`, `contracts.py`, `download.py`, `runtime/worker.py` |
| Private helper | leading underscore; never selectable | `_shared.py` |

A component category is a package whose `__init__.py` exposes the registry surface (`get, names, __getattr__, __dir__ = surface(package=__name__)`).
N2 applies inside categories; `runtime/` and `config/` hold infrastructure.
Category packages define only that registry surface [C2].
Each selectable choice with settings has its own component file and nested `Config`.

Every source package has `__init__.py`; retired directories holding only bytecode caches are rejected [N3].
Public source, docs, configs and readmes contain no private host names, network addresses, absolute workspace paths or private repository names [P1].
Cite original method sources rather than ports; the original class-count partition source is allowed in its component and partition notes [P2].

A model imports nothing from another model's directory [M1], except inverted-attention variants importing `iTransformer/InvertedTransformer.py`.
A block used by two or more models lives in `layers/`; until then it stays in its model's directory.

## Code

- Every parameter is keyword-only and annotated, and every function has a return
annotation [K1, A1].
A positional exception needs `# positional: <reason>`.
- Retained component settings keep their concrete `Config` type, including settings
passed into a strategy client [A2].
Do not widen these to `Any`; pyright must reject access to undeclared options.
Stateless components need no stored config.
- PEP 8 names, checked by ruff `N`: classes PascalCase; functions, variables and
parameters snake_case; constants UPPER_SNAKE.
Component classes keep the paper's spelling (`FedAvg`, `DLinear`).
- Config fields, parameters and local bindings use full words [D2].
Banned tokens are `n e d dim len avg lr ff enc dec hid emb seq pred std eps cfg idx attn conv tmp`, plus `fn op iter`.
Standalone `x y i j ids cap var` also need descriptive names.
The prefixes `num_`, `max_` and `min_` are accepted.
The explicit paper-symbol allowlist is `d_model` and `d_ff` (Vaswani et al., *Attention Is All You Need*) and `k` (Wu et al., *TimesNet*).
A Config declaring these symbols names the paper and symbol in its docstring [D3].
Library keyword arguments retain their API names.
- Capability constants (`TASKS`, `NEEDS`, `DIRECTION`, `PHASES`, `ABSTRACT`) are plain
literals, so the registry can read them without importing; see [Customization](../customization/index.md#where-a-component-can-be-used).
- Settings live in a nested `Config` dataclass whose first field is `name`.
- Compatibility rules live on the component as literal `REQUIRES` dictionaries
of dotted experiment paths to allowed tuples.
Code outside a component's own file does not compare its name with string literals [C1]; dispatch reads literal capability constants through the registry.
`validate(*, experiment)` handles relations between settings that cannot be expressed as allowed values.
- A quantity used as both loss and metric is one class declaring both `ROLES`.

## One name per concept

Code from papers and original repositories uses these names.
The checker fails on a deny-listed synonym [D1].

| Concept | Name | Rename on port |
|---|---|---|
| Input window length | `input_length` | `seq_len`, `lookback`, `context_length`, `context_len` |
| Forecast horizon | `output_length` | `pred_len`, `horizon`, `prediction_length` |
| Decoder warm-start length | `label_length` | `start_token_len` |
| Input-to-target gap | `offset_length` | `gap_len` |
| Input channels | `input_channels` | `enc_in`, `n_vars`, `num_features`, `c_in` |
| Output channels | `output_channels` | `c_out`, `num_targets` |
| Model width | `d_model` | `hidden_size`, `hidden_dim`, `embed_dim` |
| Number of clients | `num_clients` | `n_clients`, `client_num`, `client_num_in_total` |
| Client index | `client_id` | `cid`, `client_idx` |
| Communication round index | `current_iteration` | `round_idx`, `global_round`, `comm_round`, `round` |
| Total rounds | `iterations` | `global_rounds`, `num_rounds`, `n_comm_rounds` |
| Local epochs | `epochs` | `local_epochs`, `local_ep`, `local_n_epochs` |
| Participation fraction | `join_ratio` | `frac`, `sample_rate`, `client_fraction` |
| Learning rate | `learning_rate` | `lr`, `local_lr` |
| Batch input / target / marks | `batch_x`, `batch_y`, `x_mark`, `y_mark` | `inputs`, `targets`, `x`, `y` |

Library names we only read (such as `ast.Assign.targets`) are allowed; only names we define are checked.

## Words

In prose, comments and docstrings [W1]:

| Write | Not |
|---|---|
| directory | folder |

## Docs

- One page per component kind under `customization/`; reference material under
`reference/`; decisions under `explanation/notes.md`, without dated headings or process chatter.
- `reference/catalog.md` (one row per category), one table page per category
(`reference/components/<category>/index.md`) and one page per component under `reference/components/` are generated on every build from the registry; do not edit them.
A component's page shows its module docstring (cite the paper there), class docstring and `Config`.
- `python scripts/docs.py` builds with warnings as errors.
