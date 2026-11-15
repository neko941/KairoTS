"""Circular convolutional token embedding for TimesNet."""

from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    def __init__(self, *, input_channels: int, d_model: int) -> None:
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, *, features: Tensor) -> Tensor:
        return self.token_conv(features.permute(0, 2, 1)).transpose(1, 2)
