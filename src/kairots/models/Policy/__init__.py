"""Policy: a window's history to action logits for every step of the horizon and
every charge bin."""

from dataclasses import dataclass

from torch import Tensor, nn

from kairots.environments.Battery import Battery
from kairots.models.base import HiddenWidth


class Policy(nn.Module):
    """MLP over the flattened, window-standardized history; logits are
    `[batch, output_length, charge_bins, actions]`.

    The environment picks each step's row by the battery's current charge, so the
    policy acts closed loop; `charge_bins=1` is the open-loop plan. Standardizing
    each window keeps raw-unit inputs (the `Identity` scaler) in range.
    """

    TASKS = ("Control",)

    @dataclass(kw_only=True)
    class Config(HiddenWidth.Config):
        name: str = "Policy"
        input_length: int = 96
        output_length: int = 24
        input_channels: int = 1
        actions: int = Battery.ACTIONS
        charge_bins: int = 5

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=config.input_length * config.input_channels,
                out_features=config.hidden,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=config.hidden,
                out_features=config.output_length * config.charge_bins * config.actions,
            ),
        )

    def forward(self, *, batch_x: Tensor) -> Tensor:
        mean = batch_x.mean(dim=1, keepdim=True)
        scale = batch_x.std(dim=1, keepdim=True, unbiased=False) + 1e-5
        return self.network((batch_x - mean) / scale).reshape(
            batch_x.shape[0],
            self.config.output_length,
            self.config.charge_bins,
            self.config.actions,
        )
