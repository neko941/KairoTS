"""Two-dimensional loss contours from a two-direction grid."""

from pathlib import Path

import numpy as np


def plot(*, axis: np.ndarray, values: np.ndarray, path: Path) -> None:
    """Plot rows as the first direction and columns as the second direction."""
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    figure, axes = plt.subplots()
    try:
        axes.contourf(axis, axis, values.T)
        axes.set(xlabel="First direction", ylabel="Second direction")
        figure.savefig(path)
    finally:
        plt.close(figure)
