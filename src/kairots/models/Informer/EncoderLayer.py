"""Post-normalized ProbSparse attention and positionwise feedforward."""

from torch import Tensor, nn

from kairots.models.Informer.Attention import Attention
from kairots.models.Informer.FeedForward import FeedForward


class EncoderLayer(nn.Module):
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
        self.attention = Attention(
            d_model=d_model,
            heads=heads,
            sparse=True,
            sampling_factor=sampling_factor,
            dropout=dropout,
        )
        self.feedforward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *, features: Tensor) -> Tensor:
        hidden = self.attention_norm(
            features
            + self.dropout(
                self.attention(queries=features, keys=features, values=features)
            )
        )
        return self.feedforward_norm(hidden + self.feedforward(features=hidden))
