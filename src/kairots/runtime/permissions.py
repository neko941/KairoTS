"""Data roles available to fit, validation/support and query stages."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class DataPermissions:
    source: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    query: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        phases = (*self.source, *self.support, *self.query)
        if len(phases) != len(set(phases)):
            raise ValueError("a data phase can have only one permission role")

    @property
    def fit_phases(self) -> tuple[str, ...]:
        return (*self.source, *self.support)
