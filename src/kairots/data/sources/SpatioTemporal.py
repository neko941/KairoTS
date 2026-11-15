"""Structure kind for sensor networks recorded over time."""

from dataclasses import dataclass

from kairots.data.sources.base import Source


@dataclass(kw_only=True)
class SpatioTemporal(Source):
    """Sensor network: one long frame per network with `date`, an integer `node`
    and the feature channels; `manifest_fields` carries `num_nodes` and the road
    `graph` as `[from, to, cost]` edges. Select clients with `PerNode`.
    """

    ABSTRACT = True
