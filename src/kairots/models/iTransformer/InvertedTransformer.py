"""The iTransformer forecasting architecture, shared by its attention variants."""

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from kairots.layers.InvertedEmbedding import InvertedEmbedding
from kairots.layers.InvertedEncoder import InvertedEncoder
from kairots.layers.RevIN import RevIN
from kairots.models.base import Dropout, EmbeddingDimensions, EncoderDepth


class InvertedTransformer(nn.Module):
    """Implement iTransformer forecasting, with attention hooks for paper variants."""

    TASKS = ("Forecasting",)
    COVARIATES = ("past", "calendar")

    @dataclass(kw_only=True)
    class Config(Dropout.Config, EmbeddingDimensions.Config, EncoderDepth.Config):
        """Vaswani et al.: d_model/d_ff; iTransformer: variate-wise projection."""

        name: str = "iTransformer"
        input_length: int = 96
        output_length: int = 96
        input_channels: int = 7
        output_channels: int = 7
        calendar_channels: int = 0
        d_model: int = 512
        d_ff: int = 2048
        encoder_layers: int = 2
        attention_heads: int = 8
        activation: Literal["gelu", "relu"] = "gelu"
        use_normalization: bool = True

        def __post_init__(self) -> None:
            if (
                min(
                    self.input_length,
                    self.output_length,
                    self.input_channels,
                    self.output_channels,
                    self.d_model,
                    self.d_ff,
                    self.encoder_layers,
                    self.attention_heads,
                )
                < 1
            ):
                raise ValueError("iTransformer dimensions and counts must be positive")
            if self.output_channels > self.input_channels:
                raise ValueError("iTransformer output_channels exceeds input_channels")
            if self.d_model % self.attention_heads:
                raise ValueError("iTransformer d_model must divide evenly across heads")
            if self.calendar_channels < 0 or not 0 <= self.dropout < 1:
                raise ValueError("iTransformer calendar_channels/dropout is invalid")

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.config = config
        self.enc_embedding = InvertedEmbedding(
            input_length=config.input_length,
            d_model=config.d_model,
            dropout=config.dropout,
        )
        self.encoder = nn.ModuleList(
            InvertedEncoder(
                d_model=config.d_model,
                d_ff=config.d_ff,
                attention_heads=config.attention_heads,
                dropout=config.dropout,
                activation=config.activation,
                attention=self._attention(config=config),
            )
            for _ in range(config.encoder_layers)
        )
        self.encoder_norm = self._encoder_norm(config=config)
        self.projection = nn.Linear(config.d_model, config.output_length, bias=True)
        self.revin_layer = RevIN(input_channels=config.input_channels)

    def _attention(self, *, config: Config) -> nn.Module | None:
        return None

    def _encoder_norm(self, *, config: Config) -> nn.Module | None:
        return None

    def forward(self, *, batch_x: Tensor) -> Tensor:
        # batch_x has shape [batch_size, input_length, input_channels].
        config = self.config
        expected_length = config.input_length + (
            config.output_length if config.calendar_channels else 0
        )
        expected_channels = config.input_channels + config.calendar_channels
        if batch_x.ndim != 3 or tuple(batch_x.shape[1:]) != (
            expected_length,
            expected_channels,
        ):
            raise ValueError(
                "iTransformer needs packed input "
                f"[batch, {expected_length}, {expected_channels}]"
            )

        history = batch_x[:, : config.input_length, : config.input_channels]
        marks = (
            batch_x[:, : config.input_length, config.input_channels :]
            if config.calendar_channels
            else None
        )
        if config.use_normalization:
            history = self.revin_layer(features=history, mode="norm")
        hidden = self.enc_embedding(features=history, marks=marks)
        for layer in self.encoder:
            hidden = layer(features=hidden)
        if self.encoder_norm is not None:
            hidden = self.encoder_norm(hidden)
        forecast = self.projection(hidden).permute(0, 2, 1)[
            :, :, : config.output_channels
        ]
        if config.use_normalization:
            forecast = self.revin_layer(
                features=forecast,
                mode="denorm",
                output_channels=config.output_channels,
            )
        return forecast
