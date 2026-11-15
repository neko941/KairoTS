"""Positionwise two-layer feedforward with Informer's GELU and dropout."""

from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, *, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, *, features: Tensor) -> Tensor:
        return self.layers(features)
