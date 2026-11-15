"""Small records that point at prepared client data instead of carrying it."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class ClientRecord:
    """Where one prepared client lives and how to scale it; cheap to send to workers."""

    client_id: int
    name: str
    store: str
    values_path: str
    # Inclusive [first, last] window-start runs per split;
    # target = start + target_offset.
    windows: dict[str, tuple[tuple[int, int], ...]]
    target_offset: int
    channels: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    # Known-future calendar features appended, unscaled, after the channels.
    calendar: tuple[str, ...] = ()
    # The client's group (`Experiment.groups`), 0 without groups.
    group: int = 0
    # API fit runs may expose only source/support splits and truncate query rows.
    permissions: tuple[str, ...] | None = None
    visible_until: int | None = None
