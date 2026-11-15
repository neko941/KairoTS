"""FedProx, Li et al. (MLSys 2020): FedAvg with a proximal local objective."""

from dataclasses import dataclass

from kairots.strategies.Proximal import Proximal
from kairots.strategies.Traditional import Traditional


class FedProx(Proximal, Traditional):
    @dataclass(kw_only=True)
    class Config(Proximal.Config, Traditional.Config):
        name: str = "FedProx"
