"""XGBoost (Chen and Guestrin, KDD 2016): gradient-boosted trees mapping the
flattened input window to the flattened forecast. The booster's bytes live in a uint8
buffer, so client states, messages and checkpoints carry it; trees cannot be averaged,
so only LocalOnly and Centralized may select it. Fit with the `Estimator` learner."""

from dataclasses import dataclass

import numpy as np
import torch
import xgboost as xgb
from torch import Tensor, nn


class XGBoost(nn.Module):
    TASKS = ("Forecasting",)
    STRATEGIES = ("LocalOnly", "Centralized")
    REQUIRES = {"learner.name": ("Estimator",)}

    @dataclass(kw_only=True)
    class Config:
        name: str = "XGBoost"
        input_length: int = 96
        output_length: int = 24
        input_channels: int = 7
        num_trees: int = 50
        max_depth: int = 6
        shrinkage: float = 0.3
        seed: int = 0

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.input_length = config.input_length
        self.output_length = config.output_length
        self.input_channels = config.input_channels
        self.num_trees = config.num_trees
        self.max_depth = config.max_depth
        self.shrinkage = config.shrinkage
        self.seed = config.seed
        self.register_buffer("booster", torch.empty(0, dtype=torch.uint8))
        self._cached_booster: xgb.Booster | None = None

    def _load_from_state_dict(  # positional: PyTorch hook signature.
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = prefix + "booster"
        if key in state_dict:
            self.booster = torch.empty_like(state_dict[key], device=self.booster.device)
        self._cached_booster = None
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _booster(self) -> xgb.Booster | None:
        if self.booster.numel() == 0:
            self._cached_booster = None
            return None
        if self._cached_booster is None:
            self._cached_booster = xgb.Booster()
            self._cached_booster.load_model(
                bytearray(self.booster.cpu().numpy().tobytes())
            )
        return self._cached_booster

    def fit(self, *, batch_x: Tensor, batch_y: Tensor) -> float:
        """Add `num_trees` rounds on these windows; return their training MSE."""
        feature_values = batch_x.detach().cpu().numpy().reshape(batch_x.shape[0], -1)
        target_values = batch_y.detach().cpu().numpy().reshape(batch_y.shape[0], -1)
        training_data = xgb.DMatrix(feature_values, label=target_values)
        booster = xgb.train(
            params={
                "objective": "reg:squarederror",
                "max_depth": self.max_depth,
                "eta": self.shrinkage,
                "seed": self.seed,
            },
            dtrain=training_data,
            num_boost_round=self.num_trees,
            xgb_model=self._booster(),
        )
        self.booster = torch.from_numpy(
            np.frombuffer(booster.save_raw("ubj"), dtype=np.uint8).copy()
        ).to(device=self.booster.device)
        self._cached_booster = booster
        prediction = booster.predict(training_data)
        return float(np.mean((prediction - target_values) ** 2))

    def forward(self, *, batch_x: Tensor) -> Tensor:
        booster = self._booster()
        output_shape = (batch_x.shape[0], self.output_length, self.input_channels)
        if booster is None:
            return torch.zeros(output_shape, device=batch_x.device, dtype=batch_x.dtype)
        # CPU inference; xgboost `device="cuda"` when it matters.
        feature_values = batch_x.detach().cpu().numpy().reshape(batch_x.shape[0], -1)
        prediction = booster.predict(xgb.DMatrix(feature_values))
        prediction_tensor = torch.as_tensor(
            prediction, device=batch_x.device, dtype=batch_x.dtype
        )
        return prediction_tensor.reshape(output_shape)
