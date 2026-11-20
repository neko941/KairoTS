"""TimeKAN's Frequency Upsampling: `IFFT(Padding(FFT(x)))` along time (paper eq. 3).

Adapted from huangst21/TimeKAN@3a7c366 `frequency_interpolation` (Apache-2.0).
"""

import torch
from torch import Tensor


class FrequencyUpsampling:
    @staticmethod
    def apply(*, features: Tensor, length: int, target_length: int) -> Tensor:
        """Zero-pad the rFFT of `[rows, length, d_model]` to `target_length`, then
        invert it."""
        spectrum = torch.fft.rfft(features.transpose(1, 2), dim=2)
        padded = torch.zeros(
            spectrum.shape[0],
            spectrum.shape[1],
            target_length // 2 + 1,
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        padded[:, :, : length // 2 + 1] = spectrum
        # official code: the paper gives no scale; the code multiplies by
        # length / target_length (the paper calls the step lossless).
        upsampled = torch.fft.irfft(padded, dim=2) * (length / target_length)
        return upsampled.transpose(1, 2)
