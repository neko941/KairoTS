"""Decentralized averaging baseline: local training plus gossip over the topology.

This framework composition does not specify a standalone paper benchmark.
"""

from dataclasses import dataclass

from kairots.strategies.Decentralized import Decentralized


class DFedAvg(Decentralized):
    @dataclass(kw_only=True)
    class Config(Decentralized.Config):
        name: str = "DFedAvg"
