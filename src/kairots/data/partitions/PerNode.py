"""One client per sensor of a spatio-temporal source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(kw_only=True)
class PerNode:
    config: Config

    @dataclass(kw_only=True)
    class Config:
        name: str = "PerNode"
        nodes: tuple[int, ...] = ()

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def clients(self, *, source: Any) -> list[tuple[int, str, pl.DataFrame]]:
        clients = []
        for source_name, frame in source.clients:
            if "node" not in frame.columns:
                raise ValueError(f"{source_name}: frame has no node column")
            available = set(frame.get_column("node").unique().to_list())
            nodes = self.config.nodes or tuple(sorted(available))
            for node in sorted(nodes):
                if node not in available:
                    raise ValueError(f"{source_name}: node {node} is not in frame")
                clients.append(
                    (
                        len(clients),
                        f"{source_name}/node_{node}",
                        frame.filter(pl.col("node") == node).drop("node").sort("date"),
                    )
                )
        return clients
