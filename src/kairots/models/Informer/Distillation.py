"""Informer Eq5: length-preserving convolution then half-length max pooling."""

from torch import Tensor, nn


class Distillation(nn.Module):
    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, padding_mode="circular"
            ),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, *, features: Tensor) -> Tensor:
        return self.layers(features.transpose(1, 2)).transpose(1, 2)
