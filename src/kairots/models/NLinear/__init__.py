"""NLinear forecasting model recipe."""

from dataclasses import dataclass

from torch import Tensor, nn

from kairots.models.NLinear.Forecasting import Forecasting


class NLinear(nn.Module):
    """NLinear (Zeng et al., AAAI 2023): last-step normalization, then a linear map."""

    @dataclass(kw_only=True)
    class Config:
        name: str = "NLinear"
        input_length: int = 96
        output_length: int = 24
        input_channels: int = 7
        individual: bool = False

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.forecasting = Forecasting(
            config=Forecasting.Config(
                input_length=config.input_length,
                output_length=config.output_length,
                input_channels=config.input_channels,
                individual=config.individual,
            )
        )

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.forecasting(batch_x=batch_x)
