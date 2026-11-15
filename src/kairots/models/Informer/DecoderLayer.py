"""Masked ProbSparse self-attention followed by full encoder cross-attention."""

from torch import Tensor, nn

from kairots.models.Informer.Attention import Attention
from kairots.models.Informer.FeedForward import FeedForward


class DecoderLayer(nn.Module):
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
        self.self_attention = Attention(
            d_model=d_model,
            heads=heads,
            sparse=True,
            causal=True,
            sampling_factor=sampling_factor,
            dropout=dropout,
        )
        self.cross_attention = Attention(
            d_model=d_model, heads=heads, sparse=False, dropout=dropout
        )
        self.feedforward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.self_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *, features: Tensor, encoded: Tensor) -> Tensor:
        hidden = self.self_norm(
            features
            + self.dropout(
                self.self_attention(queries=features, keys=features, values=features)
            )
        )
        hidden = self.cross_norm(
            hidden
            + self.dropout(
                self.cross_attention(queries=hidden, keys=encoded, values=encoded)
            )
        )
        return self.feedforward_norm(hidden + self.feedforward(features=hidden))
