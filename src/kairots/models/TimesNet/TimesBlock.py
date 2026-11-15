"""Frequency-aware two-dimensional convolution block for TimesNet."""

import torch
from torch import Tensor, nn
from torch.nn import functional

from kairots.models.TimesNet.InceptionBlock import InceptionBlock


class TimesBlock(nn.Module):
    def __init__(
        self, *, d_model: int, d_ff: int, k: int, num_inception_kernels: int
    ) -> None:
        super().__init__()
        self.k = k
        self.conv = nn.ModuleList(
            [
                InceptionBlock(
                    in_channels=d_model,
                    out_channels=d_ff,
                    num_inception_kernels=num_inception_kernels,
                ),
                nn.GELU(),
                InceptionBlock(
                    in_channels=d_ff,
                    out_channels=d_model,
                    num_inception_kernels=num_inception_kernels,
                ),
            ]
        )

    def forward(self, *, features: Tensor) -> Tensor:
        batch, time_length, channels = features.shape
        spectrum = torch.fft.rfft(features, dim=1)
        amplitudes = spectrum.abs().mean(dim=0).mean(dim=-1)
        if not 1 <= self.k <= time_length // 2:
            raise ValueError(
                f"k={self.k} needs 1 to {time_length // 2} positive-frequency "
                f"bins for input length {time_length}"
            )
        frequencies = torch.topk(amplitudes[1:], self.k).indices + 1
        # TimesNet Eq. 1 specifies ceil(T / frequency); the released code floors it.
        periods = (time_length + frequencies - 1) // frequencies
        weights = spectrum.abs().mean(dim=-1)[:, frequencies]
        results = []
        for index in range(self.k):
            period = int(periods[index].item())
            length = time_length
            if length % period != 0:
                length = (length // period + 1) * period
                padding = features.new_zeros((batch, length - time_length, channels))
                padded = torch.cat((features, padding), dim=1)
            else:
                padded = features
            arranged = (
                padded.reshape(batch, length // period, period, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            convolved = self.conv[2](
                features=self.conv[1](self.conv[0](features=arranged))
            )
            restored = convolved.permute(0, 2, 3, 1).reshape(batch, -1, channels)
            results.append(restored[:, :time_length, :])
        stacked = torch.stack(results, dim=-1)
        weights = functional.softmax(weights, dim=1)[:, None, None, :].repeat(
            1, time_length, channels, 1
        )
        return torch.sum(stacked * weights, dim=-1) + features
