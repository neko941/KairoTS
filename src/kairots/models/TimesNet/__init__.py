"""TimesNet (Wu et al., ICLR 2023), implemented from TSLib's reference."""

from dataclasses import dataclass

from torch import Tensor, nn

from kairots.models.base import Dropout, EmbeddingDimensions, EncoderDepth
from kairots.models.TimesNet.Body import Body
from kairots.models.TimesNet.Classification import Classification
from kairots.models.TimesNet.Forecasting import Forecasting
from kairots.models.TimesNet.Imputation import Imputation

HEADS = {head.__name__: head for head in (Forecasting, Classification, Imputation)}


class TimesNet(nn.Module):
    """TimesNet: frequency-aware body for forecasting and classification.

    Reference: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
    Time Series Analysis," ICLR 2023; official implementation in TSLib.
    """

    COVARIATES = ("calendar",)

    @dataclass(kw_only=True)
    class Config(Dropout.Config, EmbeddingDimensions.Config, EncoderDepth.Config):
        """Wu et al., TimesNet: k is the selected frequency count."""

        name: str = "TimesNet"
        head: str = "Forecasting"
        reconstruct: bool = False
        input_length: int = 96
        output_length: int = 24
        input_channels: int = 7
        num_classes: int = 2
        k: int = 5
        num_inception_kernels: int = 6
        calendar_channels: int = 0

        def __post_init__(self) -> None:
            if self.head not in HEADS:
                raise ValueError(f"TimesNet head must be one of {tuple(HEADS)}")
            if self.calendar_channels and self.head != "Imputation":
                raise ValueError("TimesNet calendar_channels needs the Imputation head")

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.body = Body(
            config=Body.Config(
                input_channels=config.input_channels,
                d_model=config.d_model,
                d_ff=config.d_ff,
                encoder_layers=config.encoder_layers,
                k=config.k,
                num_inception_kernels=config.num_inception_kernels,
                dropout=config.dropout,
                calendar_channels=config.calendar_channels,
            )
        )
        self.head = HEADS[config.head](config=config)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.head(batch_x=batch_x, body=self.body)
