"""Three-dimensional rendering of a two-direction loss grid."""

from pathlib import Path

import numpy as np


def plot(*, axis: np.ndarray, values: np.ndarray, path: Path) -> None:
    """Use loss as height, without sampling another parameter-space direction."""
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    first, second = np.meshgrid(axis, axis, indexing="ij")
    figure = plt.figure()
    try:
        axes = figure.add_subplot(projection="3d")
        axes.plot_surface(first, second, values)
        axes.set(xlabel="First direction", ylabel="Second direction", zlabel="Loss")
        figure.savefig(path)
    finally:
        plt.close(figure)
