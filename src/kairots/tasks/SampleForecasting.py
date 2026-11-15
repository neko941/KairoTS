"""Forecast joint sample trajectories, retaining the sample axis."""

from dataclasses import dataclass

from kairots.tasks.Forecasting import Forecasting


class SampleForecasting(Forecasting):
    HEAD = "Samples"
    LOSS = "SampleCRPS"
    METRICS = ("SampleCRPS", "CRPSSum")

    @dataclass(kw_only=True)
    class Config(Forecasting.Config):
        name: str = "SampleForecasting"

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.covariates = True
