"""One ChebyshevKAN layer over the last dimension (SS et al., 2024, as used by TimeKAN).

Adapted from huangst21/TimeKAN@3a7c366 `layers/ChebyKANLayer.py` (Apache-2.0).
"""

import torch
from torch import Tensor, nn


class ChebyshevKAN(nn.Module):
    orders: Tensor

    """`out_o = sum_j sum_i Theta[j, o, i] T_i(tanh(x_j))`.

    `T_i(z) = cos(i arccos z)`.
    """

    def __init__(
        self, *, input_size: int, output_size: int, order: int, squashes: int
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.squashes = squashes
        self.coefficients = nn.Parameter(
            torch.empty(input_size, output_size, order + 1)
        )
        # official code: the paper gives no initialization;
        # N(0, 1 / (in * (order + 1))).
        nn.init.normal_(self.coefficients, mean=0.0, std=1 / (input_size * (order + 1)))
        self.register_buffer("orders", torch.arange(0, order + 1, 1))

    def forward(self, *, features: Tensor) -> Tensor:
        shape = features.shape
        flat = features.reshape(-1, shape[-1])
        flat = flat.unsqueeze(-1).expand(-1, -1, self.coefficients.shape[-1])
        # paper vs code: the paper squashes once, T_i(tanh(x)) (eq. 7); the official
        # code applies tanh twice. `squashes` is 1 (paper) or 2 (code).
        for _ in range(self.squashes):
            flat = torch.tanh(flat)
        basis = (torch.acos(flat) * self.orders).cos()
        output = torch.einsum("bid,iod->bo", basis, self.coefficients)
        return output.reshape(*shape[:-1], self.output_size)
