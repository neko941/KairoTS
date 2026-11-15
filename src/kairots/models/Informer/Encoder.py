"""Three-layer full-history and one-layer last-quarter encoder replicas."""

import torch
from torch import Tensor, nn

from kairots.models.Informer.Distillation import Distillation
from kairots.models.Informer.EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        heads: int,
        sampling_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.main = nn.ModuleList(
            EncoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                heads=heads,
                sampling_factor=sampling_factor,
                dropout=dropout,
            )
            for _ in range(3)
        )
        self.distillation = nn.ModuleList(
            Distillation(d_model=d_model) for _ in range(2)
        )
        self.replica = EncoderLayer(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            sampling_factor=sampling_factor,
            dropout=dropout,
        )
        self.main_norm = nn.LayerNorm(d_model)
        self.replica_norm = nn.LayerNorm(d_model)

    def forward(self, *, features: Tensor) -> Tensor:
        hidden = features
        for layer, distillation in zip(self.main, self.distillation):
            hidden = distillation(features=layer(features=hidden))
        hidden = self.main_norm(self.main[-1](features=hidden))
        replica = self.replica_norm(
            self.replica(features=features[:, -(features.size(1) // 4) :])
        )
        return torch.cat([hidden, replica], dim=1)
