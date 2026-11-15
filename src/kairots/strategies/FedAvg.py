"""Federated Averaging, McMahan et al. (2017)."""

from dataclasses import dataclass

from kairots.strategies.Traditional import Traditional


class FedAvg(Traditional):
    """Sample-weighted mean of client model parameters (the family default)."""

    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "FedAvg"
