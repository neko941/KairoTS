"""Decentralized proximal baseline: gossip plus a proximal local objective.

The proximal penalty follows Li et al., "Federated Optimization in Heterogeneous
Networks," MLSys 2020, arXiv:1812.06127. This framework composition anchors it on
the mixed model each client starts from; it does not specify a standalone paper
benchmark for decentralized FedProx.
"""

from dataclasses import dataclass

from kairots.strategies.Decentralized import Decentralized
from kairots.strategies.Proximal import Proximal


class DFedProx(Proximal, Decentralized):
    @dataclass(kw_only=True)
    class Config(Proximal.Config, Decentralized.Config):
        name: str = "DFedProx"
