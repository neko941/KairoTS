"""Fit a conditional diffusion model with its noise-prediction objective."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from kairots.learners.Gradient import Gradient


class Denoising(Gradient):
    REQUIRES = {"model.name": ("CSDI", "TimeGrad")}

    @dataclass(kw_only=True)
    class Config(Gradient.Config):
        name: str = "Denoising"

    def fit_epoch(
        self,
        *,
        model: Any,
        optimizer: torch.optim.Optimizer,
        batches: list[tuple[Tensor, Tensor]],
        device: torch.device,
        loss_function: Any,
        regularizer: Any,
    ) -> float:
        model.train()
        if hasattr(model, "set_patterns"):
            model.set_patterns(batches=batches)
        total = torch.zeros((), device=device)
        count = 0
        for batch_x, batch_y in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = (
                model.training_loss(
                    batch_x=batch_x.to(device=device), target=batch_y.to(device=device)
                )
                + regularizer()
            )
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(batch_y)
            count += len(batch_y)
        return float((total / max(count, 1)).item())
