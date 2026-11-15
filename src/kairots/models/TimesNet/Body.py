"""TimesNet body: embedding and frequency-aware convolution blocks."""

from dataclasses import dataclass

from torch import Tensor, nn

from kairots.models.TimesNet.DataEmbedding import DataEmbedding
from kairots.models.TimesNet.TimesBlock import TimesBlock


class Body(nn.Module):
    """`embed` maps `[B, T, C]` to `[B, T, d_model]`; `transform` runs the blocks."""

    @dataclass(kw_only=True)
    class Config:
        """Vaswani et al.: d_model/d_ff widths; Wu et al., TimesNet: frequency k."""

        input_channels: int
        d_model: int
        d_ff: int
        encoder_layers: int
        k: int
        num_inception_kernels: int
        dropout: float
        calendar_channels: int = 0

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.model = nn.ModuleList(
            TimesBlock(
                d_model=config.d_model,
                d_ff=config.d_ff,
                k=config.k,
                num_inception_kernels=config.num_inception_kernels,
            )
            for _ in range(config.encoder_layers)
        )
        self.enc_embedding = DataEmbedding(
            input_channels=config.input_channels,
            d_model=config.d_model,
            dropout=config.dropout,
            calendar_channels=config.calendar_channels,
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def embed(self, *, batch_x: Tensor, marks: Tensor | None = None) -> Tensor:
        return self.enc_embedding(features=batch_x, marks=marks)

    def transform(self, *, features: Tensor) -> Tensor:
        for block in self.model:
            features = self.layer_norm(block(features=features))
        return features

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.transform(features=self.embed(batch_x=batch_x))
