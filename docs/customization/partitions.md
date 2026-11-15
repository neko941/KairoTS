# Partitions

A partition decides which rows belong to which federated client.

**Copy:** `src/kairots/data/partitions/PerFile.py` (series) or `IID.py` (panel)

| Partition | Data | Rule |
|---|---|---|
| `PerFile` | series | one source file per client |
| `IID` | panel | samples shuffled with `seed` and dealt round-robin to `num_clients` |
| `Dirichlet` | labeled panel | classwise `Dir(alpha)` allocation; clients at the average sample count are excluded from each subsequent class |
| `ClassPerClient` | labeled panel | first eligible clients receive each class, with `balance` selecting equal or random sizes |
| `Pathological` | labeled panel | sort by label, split into equal shards, give each client `shards_per_client` random shards |
| `ExtendedDirichlet` | labeled panel | choose `classes_per_client` labels, then classwise Dirichlet allocation restricted to those clients |
| `PerNode` | spatio-temporal | one sensor node per client |

## Contract

| Member | Purpose |
|---|---|
| `Config` | nested dataclass, `name` first |
| `clients(*, source) -> list[tuple[int, str, DataFrame]]` | `(client_id, name, frame)` per client |

Client ids are dense and stable for the same source and config; the store keys the
prepared data on the partition config.

`Dirichlet` follows the heterogeneous partition of Yurochkin et al. (ICML2019,
arXiv:1905.12022v1), with the balancing step in the pinned original IBM code.
`min_samples` defaults to2; the original code uses10. It retries at most100 full
allocations before failing, and uses `default_rng(seed)`. These settings affect
client membership and belong in any reproducible experiment config.

`ExtendedDirichlet` follows Li and Lyu (NeurIPS2023, arXiv:2311.03154v3).
Its pinned original implementation supplies the paper omissions: class coverage
`max(classes_per_client * num_clients // num_classes // 5, 1)`, balancing, integer
remainder allocation and at least10 samples per client. Both rejection loops are
unbounded. Invalid coverage, insufficient samples and numerically empty draws
raise errors. The label assignment stays fixed during sample retries. It uses
`default_rng(seed)`; identical seeds do not reproduce legacy global NumPy streams.

`Pathological` follows McMahan et al. (AISTATS2017, arXiv:1602.05629v4):
`shards_per_client` defaults to2. The sample count must be divisible by
`num_clients * shards_per_client`, with nonempty shards. Within-label ties keep
source order. A seeded permutation assigns every shard once. This differs from
PFLlib's fixed-label-count partition.

`ClassPerClient` follows the pinned original PFLlib `pat` branch
(Zhang et al., JMLR2025, arXiv:2312.04992v2). `classes_per_client` limits class slots;
labels are assigned to the first eligible clients, and their samples keep source
order. `balance: true` divides each class equally, assigning remainder to the last
selected client. `balance: false` uses the original integer random-size bounds,
including the lower bound derived from batch10 and train ratio0.75. Those constants
affect allocation only; the experiment's split still controls training and test.
Uncovered labels, empty clients and impossible random-size bounds fail explicitly.

## Select it

```json
"data": {"partition": {"name": "PerFile"}}
```
