"""FedPMT: layer-wise partial model training, Wu et al., arXiv:2311.10002v1.

Reference: "Straggler-resilient Federated Learning: Tackling Computation
Heterogeneity with Layer-wise Partial Model Training in Mobile Edge Network,"
Section III and Algorithm 1. This framework assigns depths by client index:
every client trains
a suffix of the model's parameter groups with a full forward pass; client `k` trains
the last `depths[k % len(depths)]` groups (default: depths 1, ..., L)."""

from dataclasses import dataclass

from kairots.strategies.PartialTraining import PartialTraining


class FedPMT(PartialTraining):
    @dataclass(kw_only=True)
    class Config(PartialTraining.Config):
        name: str = "FedPMT"
        depths: tuple[int, ...] = ()

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def trainable(self, *, client_id: int, groups: list[str]) -> list[str]:
        depths = self.config.depths or tuple(range(1, len(groups) + 1))
        if any(depth < 1 or depth > len(groups) for depth in depths):
            raise ValueError(f"FedPMT depths must lie in [1, {len(groups)}]")
        return groups[-depths[client_id % len(depths)] :]
