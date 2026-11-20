"""TiDE (Das et al., TMLR 2023, arXiv:2304.08424): a dense MLP encoder-decoder
forecaster that reads covariates for the history and the forecast window.

Port of TSLib `models/TiDE.py`. Each target channel is forecast on its own, as in
TSLib: the history is normalized per series, the calendar features of every step
are projected by a feature encoder, and the encoder reads the history, the
projected features and (unlike TSLib, which has no past-only covariates) the
flattened past-observed covariate history. The decoder emits `decode_dimension` values
per forecast step, a temporal decoder combines each with that step's projected
features, and a linear map of the history is added as a residual.

Input layout (`tasks/Forecasting.py`): `[target_count | past | calendar]` channels over
`input_length + output_length` steps when covariates are configured, otherwise the
`[batch, input_length, channels]` history alone. Horizon values of target_count and past
covariates are NaN there and are never read. Each stacked residual block has its
own weights; TSLib repeats one block instance (`[ResBlock(...)] * n`), which
shares weights across the stack.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kairots.models.base import Dropout, EncoderDepth, HiddenWidth
from kairots.models.TiDE.ResBlock import ResBlock
from kairots.models.TiDE.stack import _stack


class TiDE(nn.Module):
    TASKS = ("Forecasting",)
    COVARIATES = ("past", "calendar")

    @dataclass(kw_only=True)
    class Config(HiddenWidth.Config, Dropout.Config, EncoderDepth.Config):
        name: str = "TiDE"
        input_length: int = 96
        output_length: int = 24
        # Source channels (target_count plus past-observed covariates) and target_count.
        input_channels: int = 7
        output_channels: int = 7
        calendar_channels: int = 0
        # TSLib ETTh1 script: d_model 256, d_ff 256, e_layers 2, d_layers 2,
        # c_out 8 (decoder output per step), dropout 0.3; feature_encode_dim 2.
        hidden: int = 256
        temporal_hidden: int = 256
        decoder_layers: int = 2
        decode_dimension: int = 8
        feature_dimension: int = 2
        dropout: float = 0.3
        # Official TiDE: per-dataset layer norm in every residual block and
        # reversible instance normalization of the history (off for some datasets).
        layer_norm: bool = True
        instance_norm: bool = True

        def __post_init__(self) -> None:
            if not 0 < self.output_channels <= self.input_channels:
                raise ValueError("TiDE needs 0 < output_channels <= input_channels")
            if min(self.encoder_layers, self.decoder_layers) < 1:
                raise ValueError("TiDE needs at least one encoder and decoder layer")

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.config = config
        length, output_length = config.input_length, config.output_length
        past = config.input_channels - config.output_channels
        features = config.feature_dimension if config.calendar_channels else 0
        self.feature_encoder = (
            ResBlock(
                input_dimension=config.calendar_channels,
                hidden=config.hidden,
                output_dimension=config.feature_dimension,
                dropout=config.dropout,
                layer_norm=config.layer_norm,
            )
            if config.calendar_channels
            else None
        )
        self.encoders = _stack(
            count=config.encoder_layers,
            input_dimension=length
            + (length + output_length) * features
            + length * past,
            hidden=config.hidden,
            output_dimension=config.hidden,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
        )
        self.decoders = _stack(
            count=config.decoder_layers,
            input_dimension=config.hidden,
            hidden=config.hidden,
            output_dimension=config.decode_dimension * output_length,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
        )
        self.temporal_decoder = ResBlock(
            input_dimension=config.decode_dimension + features,
            hidden=config.temporal_hidden,
            output_dimension=1,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
        )
        self.residual = nn.Linear(length, output_length)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        config = self.config
        length, output_length = config.input_length, config.output_length
        target_count, observed = config.output_channels, config.input_channels
        batch = batch_x.shape[0]
        history = batch_x[:, :length]
        # [batch * target_count, length]: each target channel is its own series.
        series = history[..., :target_count].transpose(1, 2).reshape(-1, length)
        if config.instance_norm:
            means = series.mean(dim=1, keepdim=True).detach()
            series = series - means
            stdev = torch.sqrt(series.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            series = series / stdev
        parts = [series]
        projected = None
        if self.feature_encoder is not None:
            calendar = batch_x[:, : length + output_length, observed:]
            projected = self.feature_encoder(features=calendar).repeat_interleave(
                target_count, dim=0
            )
            parts.append(projected.flatten(start_dim=1))
        if observed > target_count:
            past = history[..., target_count:observed].flatten(start_dim=1)
            parts.append(past.repeat_interleave(target_count, dim=0))
        hidden = torch.cat(parts, dim=1)
        for block in self.encoders:
            hidden = block(features=hidden)
        for block in self.decoders:
            hidden = block(features=hidden)
        decoded = hidden.reshape(-1, output_length, config.decode_dimension)
        if projected is not None:
            decoded = torch.cat([decoded, projected[:, length:]], dim=-1)
        output = self.temporal_decoder(features=decoded).squeeze(-1) + self.residual(
            series
        )
        if config.instance_norm:
            output = output * stdev + means
        return output.reshape(batch, target_count, output_length).transpose(1, 2)
