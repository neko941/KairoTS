"""convolution used by CSDI."""

from torch import nn


def convolution(*, source: int, destination: int) -> nn.Conv1d:
    layer = nn.Conv1d(source, destination, kernel_size=1)
    nn.init.kaiming_normal_(layer.weight)
    return layer
