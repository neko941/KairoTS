"""Fully connected topology: every node connects to every other node."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class FullyConnected:
    @dataclass(kw_only=True)
    class Config:
        name: str = "FullyConnected"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def neighbors(self, *, num_nodes: int) -> dict[int, list[int]]:
        return {
            node: [peer for peer in range(num_nodes) if peer != node]
            for node in range(num_nodes)
        }
