"""Informer (Zhou et al., AAAI 2021, arXiv:2012.07436v3).

The paper's three-layer encoder and last-quarter one-layer replica feed two
decoder layers. Learned calendar stamps accompany scalar convolution and the
cited Transformer's window-local sinusoidal positions. The latter resolves the
printed positional formula's inconsistent index range. Cross-attention is full;
encoder and masked decoder self-attention use ProbSparse sampling.

Calendar order is MonthOfYear, DayOfMonth, DayOfWeek, HourOfDay, optionally
MinuteOfHour (15-minute buckets). The native normalized calendar is inverted to
categories. Only history values reach the encoder/decoder: future values are zero
placeholders, while future calendar marks remain available.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.models.base import Dropout, EmbeddingDimensions
from kairots.models.Informer.DataEmbedding import DataEmbedding
from kairots.models.Informer.DecoderLayer import DecoderLayer
from kairots.models.Informer.Encoder import Encoder


class Informer(nn.Module):
    """Paper-first Informer with a packed native forecasting input adapter."""

    TASKS = ("Forecasting",)
    COVARIATES = ("past", "calendar")

    @dataclass(kw_only=True)
    class Config(Dropout.Config, EmbeddingDimensions.Config):
        """Vaswani et al.: d_model embedding width and d_ff feedforward width."""

        name: str = "Informer"
        input_length: int = 96
        output_length: int = 96
        label_length: int = 48
        input_channels: int = 7
        output_channels: int = 7
        calendar_channels: int = 4
        d_model: int = 512
        d_ff: int = 2048
        encoder_heads: int = 16
        decoder_heads: int = 8
        sampling_factor: int = 5

        def __post_init__(self) -> None:
            if self.input_length < 4 or not 0 < self.label_length < self.input_length:
                raise ValueError(
                    "Informer needs input_length >= 4 and "
                    "0 < label_length < input_length"
                )
            if (
                min(
                    self.output_length,
                    self.input_channels,
                    self.output_channels,
                    self.d_model,
                    self.d_ff,
                    self.encoder_heads,
                    self.decoder_heads,
                    self.sampling_factor,
                )
                < 1
            ):
                raise ValueError("Informer dimensions and counts must be positive")
            if self.output_channels > self.input_channels:
                raise ValueError(
                    "Informer output_channels cannot exceed input_channels"
                )
            if (
                self.d_model % 2
                or self.d_model % self.encoder_heads
                or self.d_model % self.decoder_heads
            ):
                raise ValueError(
                    "Informer d_model must be even and divisible by both head counts"
                )
            if self.calendar_channels not in (4, 5) or not 0 <= self.dropout < 1:
                raise ValueError(
                    "Informer needs 4 or 5 calendar channels and dropout in [0, 1)"
                )

    @staticmethod
    def validate(*, experiment: Any) -> None:
        calendar = ("MonthOfYear", "DayOfMonth", "DayOfWeek", "HourOfDay")
        if tuple(experiment.task.calendar) not in (
            calendar,
            (*calendar, "MinuteOfHour"),
        ):
            raise ValueError(
                "Informer needs calendar ordered MonthOfYear, DayOfMonth, "
                "DayOfWeek, HourOfDay, optionally MinuteOfHour"
            )

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.config = config
        self.encoder_embedding = DataEmbedding(
            input_channels=config.input_channels,
            calendar_channels=config.calendar_channels,
            d_model=config.d_model,
            max_length=config.input_length,
            dropout=config.dropout,
        )
        self.decoder_embedding = DataEmbedding(
            input_channels=config.input_channels,
            calendar_channels=config.calendar_channels,
            d_model=config.d_model,
            max_length=config.label_length + config.output_length,
            dropout=config.dropout,
        )
        self.encoder = Encoder(
            d_model=config.d_model,
            d_ff=config.d_ff,
            heads=config.encoder_heads,
            sampling_factor=config.sampling_factor,
            dropout=config.dropout,
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(
                d_model=config.d_model,
                d_ff=config.d_ff,
                heads=config.decoder_heads,
                sampling_factor=config.sampling_factor,
                dropout=config.dropout,
            )
            for _ in range(2)
        )
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.projection = nn.Linear(config.d_model, config.output_channels)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        config = self.config
        expected = (
            config.input_length + config.output_length,
            config.input_channels + config.calendar_channels,
        )
        if batch_x.ndim != 3 or tuple(batch_x.shape[1:]) != expected:
            raise ValueError(
                f"Informer needs packed input [batch, {expected[0]}, {expected[1]}]"
            )
        history = batch_x[:, : config.input_length, : config.input_channels]
        marks = batch_x[..., config.input_channels :]
        encoded = self.encoder(
            features=self.encoder_embedding(
                features=history, marks=marks[:, : config.input_length]
            )
        )
        future = history.new_zeros(
            history.size(0), config.output_length, config.input_channels
        )
        decoder_values = torch.cat([history[:, -config.label_length :], future], dim=1)
        hidden = self.decoder_embedding(
            features=decoder_values,
            marks=marks[:, config.input_length - config.label_length :],
        )
        for layer in self.decoder:
            hidden = layer(features=hidden, encoded=encoded)
        return self.projection(self.decoder_norm(hidden))[:, -config.output_length :]
