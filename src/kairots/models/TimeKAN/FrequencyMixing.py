"""TimeKAN's M-KAN learning and Frequency Mixing blocks (paper eq. 6-10).

Adapted from huangst21/TimeKAN@3a7c366 `FrequencyMixing` (Apache-2.0).
"""

from torch import Tensor, nn

from kairots.models.TimeKAN.FrequencyUpsampling import FrequencyUpsampling
from kairots.models.TimeKAN.MultiOrderKAN import MultiOrderKAN


class FrequencyMixing(nn.Module):
    """Learn each band with an M-KAN of order `b + k - i`, then rebuild the levels
    from the lowest up: `x_i = FrequencyUpsampling(x_{i+1}) + f_hat_i`."""

    def __init__(
        self, *, d_model: int, levels: int, lowest_order: int, squashes: int
    ) -> None:
        super().__init__()
        # Band i (1 = highest frequency) uses order b + k - i, so index 0 here (the
        # highest band) has the highest order and the lowest level has order b.
        self.blocks = nn.ModuleList(
            MultiOrderKAN(
                d_model=d_model,
                order=lowest_order + levels - 1 - index,
                squashes=squashes,
            )
            for index in range(levels)
        )

    def forward(self, *, bands: list[Tensor]) -> list[Tensor]:
        mixed = [self.blocks[-1](features=bands[-1])]
        for index in range(len(bands) - 2, -1, -1):
            lower = mixed[0]
            mixed.insert(
                0,
                self.blocks[index](features=bands[index])
                + FrequencyUpsampling.apply(
                    features=lower,
                    length=lower.shape[1],
                    target_length=bands[index].shape[1],
                ),
            )
        return mixed
