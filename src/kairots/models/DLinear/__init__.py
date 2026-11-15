"""DLinear model recipe: one Body, one head selected by name."""

from dataclasses import dataclass

from torch import Tensor, nn

from kairots.models.DLinear.Body import Body
from kairots.models.DLinear.Classification import Classification
from kairots.models.DLinear.Forecasting import Forecasting
from kairots.models.DLinear.Gaussian import Gaussian
from kairots.models.DLinear.Quantile import Quantile

HEADS = {
    head.__name__: head for head in (Forecasting, Classification, Quantile, Gaussian)
}


class DLinear(nn.Module):
    """DLinear (Zeng et al., AAAI 2023): decomposition + linear body, task head.

    The head sets the body's output length: Forecasting and Quantile map input_length
    to output_length and keep the horizon (Quantile projects it to `num_quantiles`
    levels); Classification keeps input_length steps, flattens and projects.
    """

    @dataclass(kw_only=True)
    class Config:
        name: str = "DLinear"
        head: str = "Forecasting"
        input_length: int = 96
        output_length: int = 24
        input_channels: int = 7
        moving_average_kernel: int = 25
        individual: bool = False
        num_classes: int = 2
        num_quantiles: int = 9

        def __post_init__(self) -> None:
            if self.head not in HEADS:
                raise ValueError(f"DLinear head must be one of {tuple(HEADS)}")

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        head_type = HEADS[config.head]
        self.body = Body(config=head_type.body(config=config))
        self.head = head_type(config=config)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.head(batch_x=batch_x, body=self.body)
