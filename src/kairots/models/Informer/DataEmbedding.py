"""Scalar convolution, approved Transformer positions and learned calendar stamps."""

from torch import Tensor, nn

from kairots.layers.PositionalEmbedding import PositionalEmbedding


class DataEmbedding(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        calendar_channels: int,
        d_model: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scalar = nn.Conv1d(
            input_channels, d_model, kernel_size=3, padding=1, padding_mode="circular"
        )
        nn.init.kaiming_normal_(
            self.scalar.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        self.position = PositionalEmbedding(d_model=d_model, max_length=max_length)
        self.calendar = nn.ModuleList(
            nn.Embedding(count, d_model)
            for count in (13, 32, 7, 24, 4)[:calendar_channels]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, *, features: Tensor, marks: Tensor) -> Tensor:
        embedded = self.scalar(features.transpose(1, 2)).transpose(1, 2)
        embedded = embedded + self.position(features=features)
        for index, (embedding, scale, offset) in enumerate(
            zip(self.calendar, (11, 30, 6, 23, 59), (1, 1, 0, 0, 0))
        ):
            categories = ((marks[..., index] + 0.5) * scale).round().long() + offset
            if index == 4:
                categories = categories // 15
            embedded = embedded + embedding(categories)
        return self.dropout(embedded)
