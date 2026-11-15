import torch
from torch import Tensor, nn

from kairots.runtime.model_statistics import model_statistics


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(96, 96)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.linear(batch_x)


def test_linear_matches_torchinfo_and_counts_true_flops() -> None:
    # torchinfo 1.8.0 reports these for Linear(96, 96) on [16, 7, 96].
    stats = model_statistics(model=_Linear(), batch_x=torch.zeros(16, 7, 96))
    assert stats["total_params"] == stats["trainable_params"] == 9312
    assert stats["mult_adds_torchinfo"] == 16 * 9312
    assert stats["mult_adds"] == 16 * 7 * 96 * 96
    assert stats["input_bytes"] == 16 * 7 * 96 * 4
    assert stats["forward_backward_bytes"] == 2 * 16 * 7 * 96 * 4
    assert stats["param_bytes"] == 9312 * 4


def test_convolution_counts_every_output_position() -> None:
    class Convolution(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convolution = nn.Conv1d(7, 4, kernel_size=3)

        def forward(self, *, batch_x: Tensor) -> Tensor:
            return self.convolution(batch_x)

    stats = model_statistics(model=Convolution(), batch_x=torch.zeros(2, 7, 10))
    # Output [2, 4, 8]: (84 weights + 4 biases) x 2 x 8.
    assert stats["mult_adds_torchinfo"] == 88 * 2 * 8
