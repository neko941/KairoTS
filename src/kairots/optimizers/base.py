"""The optimizer regularization option shared by the optimizer implementations."""

from dataclasses import dataclass


class Optimizer:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Optimizer"
        weight_decay: float = 0.0


class UpdateBudget:
    @dataclass(kw_only=True)
    class Config:
        name: str = "UpdateBudget"
        steps: int = 1
