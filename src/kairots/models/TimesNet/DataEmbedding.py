"""Value and positional embeddings for TimesNet."""

from torch import Tensor, nn

from kairots.layers.PositionalEmbedding import PositionalEmbedding
from kairots.models.TimesNet.TokenEmbedding import TokenEmbedding


class DataEmbedding(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        d_model: int,
        dropout: float,
        calendar_channels: int = 0,
    ) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(
            input_channels=input_channels, d_model=d_model
        )
        # TSLib's timeF temporal embedding exists only when calendar marks are given.
        self.temporal_embedding = (
            nn.Linear(calendar_channels, d_model, bias=False)
            if calendar_channels
            else None
        )
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, *, features: Tensor, marks: Tensor | None = None) -> Tensor:
        embedded = self.value_embedding(features=features) + self.position_embedding(
            features=features
        )
        if marks is not None:
            if self.temporal_embedding is None:
                raise ValueError("calendar marks need TimesNet calendar_channels > 0")
            embedded = embedded + self.temporal_embedding(marks)
        return self.dropout(embedded)
