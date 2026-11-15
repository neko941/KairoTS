"""Gradient fitting with the N1 optimizer and numerical switches."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.learners.base import Learner, Learning
from kairots.registry import get


@dataclass(kw_only=True)
class Gradient:
    @dataclass(kw_only=True)
    class Config(Learning.Config, Learner.Config):
        name: str = "Gradient"
        precision: str = "fp32"
        fused_optimizer: bool = True
        # Gradient 2-norm above which every update is rescaled; 0 never clips.
        max_gradient_norm: float = 0.0

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def optimizer(self, *, model: nn.Module) -> torch.optim.Optimizer:
        optimizer_type = get(category="optimizers", name=self.config.optimizer.name)
        return optimizer_type(config=self.config.optimizer).build(
            parameters=model.parameters(),
            learning_rate=self.config.learning_rate,
            fused=self.config.fused_optimizer,
        )

    def fit_epoch(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: list[tuple[Tensor, Tensor]],
        device: torch.device,
        loss_function: Any,
        regularizer: Any,
    ) -> float:
        model.train()
        loss_sum = torch.zeros((), device=device)
        count = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device=device, non_blocking=device.type == "cuda")
            batch_y = batch_y.to(device=device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=self.config.precision == "bf16" and device.type == "cuda",
            ):
                prediction = model(batch_x=batch_x)
                loss = (
                    loss_function(prediction=prediction, target=batch_y) + regularizer()
                )
            loss.backward()
            if self.config.max_gradient_norm:
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.config.max_gradient_norm
                )
            optimizer.step()
            loss_sum += loss.detach() * batch_y.numel()
            count += batch_y.numel()
        return float((loss_sum / max(count, 1)).item())
