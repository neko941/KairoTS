"""One-dimensional loss curves and checkpoint interpolation."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from kairots.analysis.landscape.common import mean_loss


def interpolation(
    *,
    model: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    loss_function: Any,
    end: dict[str, Tensor],
    points: int,
) -> Tensor:
    """Loss at `(1 - alpha) * model + alpha * end` for `points` alphas in [0, 1].

    A parameter missing from `end` stays at the model's value.
    """
    parameters = dict(model.named_parameters())
    original = {name: value.detach().clone() for name, value in parameters.items()}
    result = torch.empty(points)
    try:
        with torch.no_grad():
            for index, alpha in enumerate(torch.linspace(0.0, 1.0, points)):
                for name, parameter in parameters.items():
                    target = end.get(name, original[name]).to(parameter)
                    parameter.copy_(torch.lerp(original[name], target, alpha))
                result[index] = mean_loss(
                    model=model, batches=batches, loss_function=loss_function
                )
    finally:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(original[name])
    return result


def plot(*, axis: np.ndarray, values: np.ndarray, path: Path) -> None:
    """Plot loss along one parameter-space direction."""
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    figure, axes = plt.subplots()
    try:
        axes.plot(axis, values)
        axes.set(xlabel="Direction coordinate", ylabel="Loss")
        figure.savefig(path)
    finally:
        plt.close(figure)
