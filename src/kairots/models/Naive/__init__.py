"""Naive (last value) forecast, the persistence baseline of TSLib's short-term
forecasting: every horizon step repeats the last observed step. It has no parameters,
so select it with the `Stateless` learner."""

from dataclasses import dataclass

from torch import Tensor, nn


class Naive(nn.Module):
    TASKS = ("Forecasting",)

    @dataclass(kw_only=True)
    class Config:
        name: str = "Naive"
        output_length: int = 24

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.output_length = config.output_length

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return batch_x[:, -1:, :].expand(-1, self.output_length, -1)
