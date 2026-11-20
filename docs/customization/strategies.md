# Strategies

A strategy owns the federated round: which clients train, what they receive, how they train, and how results combine.
Strategies inherit from a **family** that fixes the round shape; a method overrides only what differs.

**Copy:** `src/kairots/strategies/FedAvg.py` (family: `Traditional.py`)

## Families

| Family | Round shape |
|---|---|
| `Traditional` | server selects clients, broadcasts the global model, aggregates |
| `NonFederated` | every client trains and keeps its own model; nothing is aggregated or billed |
| `Decentralized` | every client keeps its own model and averages with its topology neighbours; no server |
| `PartialTraining` | full forward, but each client trains and uploads only the parameter groups `trainable(*, client_id, groups)` returns; the server averages each parameter over the clients that trained it (`FedPMT`) |
| `Quantized` | each client uploads its whole update through a lossy codec (`quantize`, `encoded_bytes`); the server adds the mean decoded update (`FedPAQ`) |
| `Personalized` | federated global model plus a personal model per client in `ClientState.extra`; clients are scored with their personal model |
| `Proximal` | local objective only: adds `mu / 2 * ||w - w_start||^2`; combine it with a family above |

## Combining families

A method may inherit from several families when they own different hooks (`base.HOOKS`): `FedProx(Proximal, Traditional)`, `DFedProx(Proximal, Decentralized)`.
If two parent families override the same hook, the class fails at import unless it defines that hook itself, so a clash never resolves silently by method order.

## Contract

| Member | Purpose |
|---|---|
| `Config` | subclass the family's `Config`; set `name` |
| `SOURCE_RUN = True` | resolve an optional source training run, require a complete compatible checkpoint, and score with a `cross_` prefix (`ZeroShot`) |
| `setup(*, experiment)` | server-side preparation before round 0 (default nothing; `FedTREND` keeps the model config and window lengths for its synthetic data) |
| `select_clients(*, client_ids, current_iteration, seed)` | family default: seeded `join_ratio` sample |
| `build_downlink(*, global_params, client_id) -> Message` | what a client receives |
| `initial_state(*, global_params) -> ClientState` | client state before round 0 |
| `client_parameters(*, global_params, state) -> dict[str, Tensor]` | parameters a client is evaluated with |
| `personal_model(*, model, experiment) -> nn.Module` | the module personalization scores, built once from the client model and filled by `client_parameters` (default: the model; `MoEFL`: its gated mixture) |
| `split_state(*, parameters) -> (uplink, kept)` | parameters sent and kept after training |
| `uplink_scalars(*, tensors, train_samples) -> dict[str, float]` | scalars sent with the uplink tensors (default none; `Traditional`: the sample count when tensors are sent) |
| `exchange(*, results) -> list[Message]` | peer step after local training; messages are billed as peer traffic |
| `Client.fit(...)`, `Client.proximal_term(...)` | local training; the term anchors on the model the client started the round from |
| `client_update(*, worker, job) -> Result` | builds the `Client` (with its options) and runs it |
| `aggregate(*, uplinks, global_params) -> dict[str, Tensor]` | new global parameters from the clients' uplink `Message`s |
| `broadcast(*, uplinks, global_params) -> list[Message]` | optional post-aggregation server messages, billed as downlink (`FedRidge` sends its fitted weights here) |

Personal parameters live in `ClientState.model_state`.
Optimizer state and other client state also travel in `ClientState`, so Serial and Ray give identical results.
Traditional weighted aggregation requires positive sample counts for parameter updates.
Parameter-free models with empty client states leave the empty global state unchanged even when their stateless learner reports zero samples.

## Billed traffic

Each round reports `downlink_bytes`, `uplink_bytes` and `peer_bytes`.
Only `Message`s are billed: tensor bytes plus 4 bytes per scalar, or `encoded_bytes` when a codec sets it.
A message whose sender is its receiver bills 0.
`ClientState` (personal models, optimizer state, `extra`) never crosses the network and is never billed.
What crosses differs by method, so a new strategy must bill exactly what its paper sends:

| Strategy | Downlink | Uplink | Peer |
|---|---|---|---|
| `FedAvg`, `FedProx`, `FLwF2`, `Ditto` | global model | model + sample count | |
| `FedBABU` | body | body + sample count | |
| `SCAFFOLD` | model + server control variate | model + client control variate | |
| `APFL` | global model | global model | |
| `FML` | meme model | meme model | |
| `FedPMT` | global model | trained groups only | |
| `MoEFL` | global model | model + sample count (FedAvg rounds, opt-in clients); nothing in the mixture round | |
| `FedPAQ` | global model | encoded update (`encoded_bytes`) | |
| `FedTREND` | global model, plus `D_ct` (`client_synthetic_size × (input + output)` floats) once after each construction | model (+ sample count with `reference: "paper"`) | |
| `FedRidge` | fitted weight, after aggregation (`broadcast`) | Gram triangle + moment | |
| `SplitNN` | | cut-layer activations and gradients (`encoded_bytes`) | |
| `DFedAvg`, `DFedProx` | | | full model per directed edge |
| `LocalOnly`, `Transfer`, `Centralized`, `ZeroShot` | | | |

- **What is on the wire.
The uplink `Message` is the client's whole transmission: the tensors from `split_state` plus the scalars from `uplink_scalars(*, tensors, train_samples)`.
A method returns there whatever its paper sends beside the tensors (a sample count, a local step count, a score), and nothing else.
`Traditional` sends `train_samples` because its average is sample-weighted; `SCAFFOLD`, `APFL`, `Quantized` and `Heterogeneous` average unweighted and send none.
A scalar the count does not give (a local step count) is added to `result.uplink.scalars` in `client_update`.
- **The server reads only the wire.
`aggregate` and `broadcast` receive the uplink `Message`s, never `Result`s, so `train_samples`, `metrics` and `state` (which do not cross the network) are out of their reach.
`Traditional` raises `KeyError` when the count it weights by was not sent.
- **Only billable values travel.
A `Message` accepts tensors and numeric scalars only (4 bytes each) and raises `TypeError` on anything else; send a mask as a `bool` or `uint8` tensor and a wider number as a tensor of its dtype.
- **Rebuilt uplinks.
A strategy that replaces the uplink in `client_update` (`Quantized`, `PartialTraining`, `SplitNN`) bills only what it puts in the new `Message`.
- **Extra server messages.
Messages from `broadcast` bill as downlink, and messages from `exchange` bill as peer traffic.
`exchange` is peer code: it mixes what the peers send each other and bills one message per directed edge.
- **Per-client records.
`clients.jsonl` gives each client's `downlink_bytes`, `uplink_bytes` and `peer_bytes`; messages to or from `client:<id>` count for that client, and other parties' messages count only in the round totals.

## Select it

```json
"strategy": {"name": "FedAvg", "iterations": 100, "join_ratio": 0.1}
```

## Test it

Test `aggregate` on hand-made uplink `Message`s with plain tensors; see `test_fedavg_is_exact_sample_weighted_mean`.
