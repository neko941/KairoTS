"""TimeKAN (Huang et al., ICLR 2025), KAN-based frequency decomposition forecasting.

Reference: huangst21/TimeKAN@3a7c366a9e8547fd8840c5d27f25ee3e30615e33 (Apache-2.0).
Each variate is forecast independently: average pooling builds `k` levels, and
`encoder_layers` rounds of Cascaded Frequency Decomposition, M-KAN learning and
Frequency Mixing rebuild them; linear layers map the highest level to the horizon.
`reference: "paper"` follows the published paper where it and the official code
disagree; `reference: "code"` follows the code (the paper's tables came from it).
Lines whose behavior comes from the code are commented `official code:` or
`paper vs code:` in this directory.
"""

from dataclasses import dataclass

from torch import Tensor, nn

from kairots.layers.RevIN import RevIN
from kairots.models.base import Dropout, EncoderDepth
from kairots.models.TimeKAN.FrequencyDecomposition import FrequencyDecomposition
from kairots.models.TimeKAN.FrequencyMixing import FrequencyMixing
from kairots.models.TimeKAN.ValueEmbedding import ValueEmbedding

REFERENCES = ("paper", "code")


class TimeKAN(nn.Module):
    """TimeKAN forecasting recipe."""

    TASKS = ("Forecasting",)

    @dataclass(kw_only=True)
    class Config(EncoderDepth.Config, Dropout.Config):
        """TimeKAN settings; `d_model` is the paper's embedding dimension D (Vaswani
        et al., *Attention Is All You Need*, d_model)."""

        name: str = "TimeKAN"
        head: str = "Forecasting"
        input_length: int = 96
        output_length: int = 96
        input_channels: int = 7
        d_model: int = 16
        # k - 1 pooled levels below the input (official `down_sampling_layers`).
        down_sampling_layers: int = 2
        # Average-pooling window d (official `down_sampling_window`).
        down_sampling_window: int = 2
        # Lowest Chebyshev order b (official `begin_order`, default 1).
        lowest_order: int = 1
        reference: str = "paper"

        def __post_init__(self) -> None:
            if self.head != "Forecasting":
                raise ValueError("TimeKAN has only a Forecasting head")
            if self.reference not in REFERENCES:
                raise ValueError(f"TimeKAN reference must be one of {REFERENCES}")
            if self.down_sampling_layers < 1 or self.down_sampling_window < 2:
                raise ValueError("TimeKAN needs >= 1 pooled level and a window >= 2")
            # paper vs code: the paper pads before pooling (eq. 1) without saying how;
            # the code pools without padding and floors. Paper mode accepts only
            # lengths the pooling divides, where both agree.
            if self.reference == "paper" and self.input_length % (
                self.down_sampling_window**self.down_sampling_layers
            ):
                raise ValueError(
                    "TimeKAN paper mode needs input_length divisible by "
                    "down_sampling_window ** down_sampling_layers (padding unspecified)"
                )

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.config = config
        code = config.reference == "code"
        levels = config.down_sampling_layers + 1
        squashes = 2 if code else 1
        self.pool = nn.AvgPool1d(config.down_sampling_window)
        # official code: every level is RevIN-normalized with its own affine (paper
        # silent); only the input level's statistics undo the forecast.
        self.normalizations = [
            RevIN(input_channels=config.input_channels, affine=True, detach_scale=True)
            for _ in range(levels)
        ]
        self.normalization_modules = nn.ModuleList(self.normalizations)
        self.embedding = ValueEmbedding(
            d_model=config.d_model, dropout=config.dropout, convolutional=code
        )
        self.decompositions = nn.ModuleList(
            FrequencyDecomposition() for _ in range(config.encoder_layers)
        )
        self.mixings = nn.ModuleList(
            FrequencyMixing(
                d_model=config.d_model,
                levels=levels,
                lowest_order=config.lowest_order,
                squashes=squashes,
            )
            for _ in range(config.encoder_layers)
        )
        # official code: the paper's "simple linear layer" (eq. 11) is a time
        # projection input_length -> output_length, then d_model -> 1.
        self.time_projection = nn.Linear(config.input_length, config.output_length)
        self.channel_projection = nn.Linear(config.d_model, 1)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        batch, _, channels = batch_x.shape
        series = batch_x.transpose(1, 2)
        levels = [batch_x]
        for _ in range(self.config.down_sampling_layers):
            series = self.pool(series)
            levels.append(series.transpose(1, 2))
        embedded = []
        for normalization, level in zip(self.normalizations, levels, strict=True):
            level = normalization(features=level, mode="norm")
            rows = level.transpose(1, 2).reshape(batch * channels, -1, 1)
            embedded.append(self.embedding(features=rows))
        for decomposition, mixing in zip(
            self.decompositions, self.mixings, strict=True
        ):
            embedded = mixing(bands=decomposition(levels=embedded))
        highest = self.time_projection(embedded[0].transpose(1, 2)).transpose(1, 2)
        forecast = self.channel_projection(highest).reshape(
            batch, channels, self.config.output_length
        )
        return self.normalizations[0](features=forecast.transpose(1, 2), mode="denorm")
