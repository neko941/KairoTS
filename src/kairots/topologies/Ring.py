"""Ring topology: each node connects to its cyclic neighbours."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class Ring:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Ring"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def neighbors(self, *, num_nodes: int) -> dict[int, list[int]]:
        if num_nodes < 2:
            return {node: [] for node in range(num_nodes)}
        return {
            node: sorted({(node - 1) % num_nodes, (node + 1) % num_nodes})
            for node in range(num_nodes)
        }
