"""Unconditional synthesis: learn the distribution of `L`-step windows and sample new
ones from noise, with a denoising diffusion model (DDPM, Ho et al., 2020).

The split gives target windows (`input_length = 0`, `output_length = L`).
`prepare` draws a diffusion step `t` and Gaussian noise per window:

- model input: the noised window `sqrt(abar_t) x + sqrt(1 - abar_t) noise`, with `t`
  appended as a last channel (constant over the window);
- target: the noise, so the MSE loss is DDPM's simplified objective.

`sample` runs the reverse process from pure noise: each step predicts the clean
window from the noise estimate, clamps it to `[-clip, clip]` scaled units (`clip = 0`
turns it off), and draws from the posterior `q(x_{t-1} | x_t, x_0)` (TimeDP's
`p_sample`). Betas rise linearly from `beta_start` to `beta_end` over `diffusion_steps`.
Training draws continue the job's generator; scoring draws restart from `seed` for
every client, so scoring noise and samples are fixed. Scoring belongs to the
`Generative` protocol.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kairots.models.base import Diffusion


@dataclass(kw_only=True)
class Generation:
    HEAD = "Generation"
    RECONSTRUCTS = True
    LOSS = "MSE"
    METRICS = (
        "MMD",
        "Wasserstein",
        "ACFError",
        "CorrelationError",
        "DiscriminativeScore",
        "TSTR",
        "DCR",
    )
    PROTOCOL = "Generative"

    @dataclass(kw_only=True)
    class Config(Diffusion.Config):
        name: str = "Generation"
        clip: float = 5.0
        seed: int = 0

        def __post_init__(self) -> None:
            if self.diffusion_steps < 1:
                raise ValueError("Generation diffusion_steps must be >= 1")
            if not 0.0 < self.beta_start <= self.beta_end < 1.0:
                raise ValueError("Generation needs 0 < beta_start <= beta_end < 1")
            if self.clip < 0:
                raise ValueError("Generation clip must be >= 0")

    def __init__(self, *, config: Config) -> None:
        self.config = config
        self.betas = torch.linspace(
            config.beta_start,
            config.beta_end,
            config.diffusion_steps,
            dtype=torch.float64,
        )
        self.alpha_bars = torch.cumprod(1 - self.betas, dim=0)

    @staticmethod
    def _input(*, noised: Tensor, steps: Tensor) -> Tensor:
        step_channel = steps.to(dtype=noised.dtype, device=noised.device)
        return torch.cat(
            [noised, step_channel[:, None, None].expand(-1, noised.shape[1], 1)],
            dim=-1,
        )

    def prepare(
        self,
        *,
        batch_x: Tensor,
        batch_y: Tensor,
        output_length: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if generator is None:
            raise ValueError("Generation needs a generator to draw noise")
        steps = torch.randint(
            self.config.diffusion_steps, (batch_y.shape[0],), generator=generator
        )
        noise = torch.randn(batch_y.shape, generator=generator, dtype=batch_y.dtype)
        alpha_bar = self.alpha_bars[steps].to(dtype=batch_y.dtype)[:, None, None]
        noised = alpha_bar.sqrt() * batch_y + (1 - alpha_bar).sqrt() * noise
        return self._input(noised=noised, steps=steps), noise

    def sample(
        self,
        *,
        model: nn.Module,
        count: int,
        shape: tuple[int, int],
        generator: torch.Generator,
        device: torch.device,
    ) -> Tensor:
        """`count` windows of `shape` (steps, channels) drawn from the model."""
        windows = torch.randn((count, *shape), generator=generator).to(device=device)
        for step in reversed(range(self.config.diffusion_steps)):
            beta = float(self.betas[step])
            alpha_bar = float(self.alpha_bars[step])
            previous = float(self.alpha_bars[step - 1]) if step else 1.0
            noise = model(
                batch_x=self._input(noised=windows, steps=torch.full((count,), step))
            )
            clean = (windows - (1 - alpha_bar) ** 0.5 * noise) / alpha_bar**0.5
            if self.config.clip:
                clean = clean.clamp(-self.config.clip, self.config.clip)
            windows = (
                previous**0.5 * beta * clean
                + (1 - beta) ** 0.5 * (1 - previous) * windows
            ) / (1 - alpha_bar)
            if step:
                variance = beta * (1 - previous) / (1 - alpha_bar)
                windows = windows + variance**0.5 * torch.randn(
                    windows.shape, generator=generator
                ).to(device=device)
        return windows
