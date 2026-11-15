"""Composition of source datasets with stable client provenance."""

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from kairots.data.sources.base import Source


@dataclass(kw_only=True)
class Composite(Source):
    ABSTRACT = True

    @staticmethod
    def join(*, members: Sequence[Source]) -> list[tuple[int, str, pl.DataFrame]]:
        return [
            (client_id, f"{source.name}/{client_name}", frame)
            for client_id, (source, client_name, frame) in enumerate(
                (source, name, frame)
                for source in members
                for name, frame in source.clients
            )
        ]
