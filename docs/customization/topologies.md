# Topologies

A topology maps each node index to its sorted, undirected peer indices. Decentralized
strategies use this map for one Metropolis weighted gossip step after local training.

**Copy:** `src/kairots/topologies/Ring.py`

## Contract

| Member | Purpose |
|---|---|
| `Config` | nested dataclass, `name` first |
| `neighbors(*, num_nodes) -> dict[int, list[int]]` | peer indices for each node; never itself |

## Select it

```json
"strategy": {"name": "DFedAvg", "topology": "Ring"}
```
