"""ResBlock used by TiDE."""

from torch import Tensor, nn


class ResBlock(nn.Module):
    """TSLib ResBlock: Linear-ReLU-Linear-Dropout, skip, optional LayerNorm."""

    def __init__(
        self,
        *,
        input_dimension: int,
        hidden: int,
        output_dimension: int,
        dropout: float,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dimension),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(input_dimension, output_dimension)
        self.norm = nn.LayerNorm(output_dimension) if layer_norm else nn.Identity()

    def forward(self, *, features: Tensor) -> Tensor:
        return self.norm(self.dense(features) + self.skip(features))
