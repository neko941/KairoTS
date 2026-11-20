"""TimeKAN's M-KAN block: a channel ChebyshevKAN plus a depthwise temporal convolution.

Adapted from huangst21/TimeKAN@3a7c366 `models/TimeKAN.py` (`M_KAN`, `BasicConv`;
Apache-2.0).
"""

from torch import Tensor, nn

from kairots.models.TimeKAN.ChebyshevKAN import ChebyshevKAN


class MultiOrderKAN(nn.Module):
    """`f_hat = KAN(f, order) + DepthwiseConv(f)` on `[rows, length, d_model]`.

    Paper eq. 6-9.
    """

    # official code: the paper does not give the kernel; `BasicConv(kernel_size=3)`.
    KERNEL = 3

    def __init__(self, *, d_model: int, order: int, squashes: int) -> None:
        super().__init__()
        self.kan = ChebyshevKAN(
            input_size=d_model, output_size=d_model, order=order, squashes=squashes
        )
        # official code: stride 1, padding kernel // 2, no bias (paper silent).
        self.convolution = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=self.KERNEL,
            padding=self.KERNEL // 2,
            groups=d_model,
            bias=False,
        )

    def forward(self, *, features: Tensor) -> Tensor:
        temporal = self.convolution(features.transpose(-1, -2)).transpose(-1, -2)
        return self.kan(features=features) + temporal
