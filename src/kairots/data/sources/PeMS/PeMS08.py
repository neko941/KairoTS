"""PeMS08 (San Bernardino, July to August 2016): 170 loop sensors, flow,
occupancy and speed every 5 minutes, 295 road edges.
"""

from dataclasses import dataclass
from datetime import datetime

from kairots.data.sources.PeMS import PeMS


class PeMS08(PeMS):
    TASKS = ("Forecasting",)
    DISTRICT = "PEMS08"
    START = datetime(2016, 7, 1)
    SHA256 = {
        "npz": "e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075",
        "csv": "859d68e7583ed4e6016588eaaca494d3ece286dae361614c5c1833017f87eaa7",
    }

    @dataclass(kw_only=True)
    class Config(PeMS.Config):
        name: str = "PeMS08"
