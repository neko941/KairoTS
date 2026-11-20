"""TimeKAN's Cascaded Frequency Decomposition block (paper eq. 3-4).

Adapted from huangst21/TimeKAN@3a7c366 `FrequencyDecomp` (Apache-2.0).
"""

from torch import Tensor, nn

from kairots.models.TimeKAN.FrequencyUpsampling import FrequencyUpsampling


class FrequencyDecomposition(nn.Module):
    """Levels `[x_1, ..., x_k]` (highest first) to bands `[f_1, ..., f_{k-1}, x_k]`.

    `f_i = x_i - FrequencyUpsampling(x_{i+1})`; the lowest level is its own band.
    """

    def forward(self, *, levels: list[Tensor]) -> list[Tensor]:
        bands = [
            levels[index]
            - FrequencyUpsampling.apply(
                features=levels[index + 1],
                length=levels[index + 1].shape[1],
                target_length=levels[index].shape[1],
            )
            for index in range(len(levels) - 1)
        ]
        return [*bands, levels[-1]]
