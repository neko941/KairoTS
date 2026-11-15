"""Assign one source file/client record to each logical client."""

from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class PerFile:
    config: object

    @dataclass(kw_only=True)
    class Config:
        name: str = "PerFile"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def clients(self, *, source: Any) -> list[tuple[int, str, Any]]:
        return [
            (client_id, name, frame)
            for client_id, (name, frame) in enumerate(source.clients)
        ]
