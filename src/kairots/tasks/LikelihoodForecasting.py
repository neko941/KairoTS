"""Forecast a Gaussian likelihood at every target position."""

from dataclasses import dataclass

from kairots.tasks.Forecasting import Forecasting


class LikelihoodForecasting(Forecasting):
    HEAD = "Gaussian"
    RECONSTRUCTS = False
    LOSS = "GaussianNLL"
    METRICS = ("GaussianNLL",)

    @dataclass(kw_only=True)
    class Config(Forecasting.Config):
        name: str = "LikelihoodForecasting"
