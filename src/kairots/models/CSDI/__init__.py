"""Conditional score-based imputation with temporal and feature attention."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from kairots.models.base import ConditionalDiffusion, ResidualStack
from kairots.models.CSDI.convolution import convolution
from kairots.models.CSDI.Residual import Residual

MAX_PARALLEL_SAMPLES = 128


class CSDI(nn.Module):
    """Tashiro et al., NeurIPS 2021, arXiv:2107.03502v2."""

    TASKS = ("ProbabilisticImputation",)
    REQUIRES = {
        "task.calendar": ((), ("ElapsedHours",)),
        "learner.name": ("Denoising",),
        "learner.precision": ("fp32",),
    }
    MASKS = True
    COVARIATES = ("calendar",)

    @dataclass(kw_only=True)
    class Config(ConditionalDiffusion.Config, ResidualStack.Config):
        name: str = "CSDI"
        input_length: int = 24
        output_length: int = 24
        input_channels: int = 7
        heads: int = 8
        diffusion_steps: int = 50
        beta_end: float = 0.5
        target_strategy: str = "random"
        target_pattern: tuple[tuple[bool, ...], ...] = ()
        seed: int = 0

        def __post_init__(self) -> None:
            if self.width < 1 or self.heads < 1 or self.width % self.heads:
                raise ValueError("CSDI width must be divisible by heads")
            if (
                min(
                    self.layers,
                    self.diffusion_steps,
                    self.samples,
                    self.input_channels,
                    self.input_length,
                )
                < 1
            ):
                raise ValueError("CSDI dimensions and counts must be positive")
            if not 0 < self.beta_start <= self.beta_end < 1:
                raise ValueError("CSDI beta schedule must lie in (0, 1)")
            if self.target_strategy not in ("random", "historical", "mix", "test"):
                raise ValueError("unknown CSDI target strategy")
            if self.target_strategy == "test" and not self.target_pattern:
                raise ValueError("test target strategy needs target_pattern")
            if self.target_pattern and (
                len(self.target_pattern) != self.input_length
                or any(len(row) != self.input_channels for row in self.target_pattern)
            ):
                raise ValueError(
                    "target_pattern must match time and feature dimensions"
                )

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.beta: Tensor
        self.alpha: Tensor
        self.step_table: Tensor
        self.config = config
        beta = torch.linspace(
            config.beta_start**0.5, config.beta_end**0.5, steps=config.diffusion_steps
        ).square()
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", (1 - beta).cumprod(dim=0))
        angles = torch.arange(1, config.diffusion_steps + 1)[:, None] * 10 ** (
            torch.arange(64)[None, :] / 63 * 4
        )
        self.register_buffer(
            "step_table",
            torch.cat([angles.sin(), angles.cos()], dim=-1),
            persistent=False,
        )
        self.step_projection = nn.Sequential(
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU()
        )
        self.feature_embedding = nn.Embedding(config.input_channels, 16)
        self.input_projection = convolution(source=2, destination=config.width)
        self.blocks = nn.ModuleList(
            Residual(width=config.width, heads=config.heads)
            for _ in range(config.layers)
        )
        self.output_projection = nn.Sequential(
            convolution(source=config.width, destination=config.width),
            nn.ReLU(),
            convolution(source=config.width, destination=1),
        )
        final = self.output_projection[-1]
        assert isinstance(final, nn.Conv1d)
        nn.init.zeros_(final.weight)
        self.patterns: Tensor | None = None

    def set_patterns(self, *, batches: list[tuple[Tensor, Tensor]]) -> None:
        if self.config.target_strategy in ("historical", "mix"):
            self.patterns = torch.cat(
                [
                    (
                        features[
                            ...,
                            self.config.input_channels : 2 * self.config.input_channels,
                        ]
                        > 0
                    )
                    | target_values.isfinite()
                    for features, target_values in batches
                ]
            ).cpu()

    def conditional_mask(self, *, observed: Tensor) -> Tensor:
        conditional = observed.contiguous().clone()
        for index in range(len(observed)):
            strategy = self.config.target_strategy
            if strategy == "mix":
                strategy = "historical" if torch.rand(()).item() < 0.5 else "random"
            if strategy == "random":
                positions = observed[index].flatten().nonzero().flatten()
                count = int(len(positions) * torch.rand(()).item() + 0.5)
                choices = torch.randperm(len(positions), device=observed.device)[:count]
                conditional[index].view(-1)[positions[choices]] = False
            elif strategy == "test":
                conditional[index] &= torch.tensor(
                    self.config.target_pattern, device=observed.device, dtype=torch.bool
                )
            else:
                if self.patterns is None or not len(self.patterns):
                    raise ValueError("historical masks need training patterns")
                choice = int(torch.randint(len(self.patterns), size=()).item())
                conditional[index] &= self.patterns[choice].to(device=observed.device)
        return conditional

    def predict_noise(
        self,
        *,
        noisy: Tensor,
        values: Tensor,
        mask: Tensor,
        step: Tensor,
        times: Tensor | None = None,
    ) -> Tensor:
        batch, length, channels = values.shape
        angles = torch.arange(length, device=values.device)[:, None] / 10000 ** (
            torch.arange(64, device=values.device)[None, :] * 2 / 128
        )
        if times is not None:
            angles = times[..., None] / 10000 ** (
                torch.arange(64, device=values.device)[None, None, :] * 2 / 128
            )
            temporal = torch.cat([angles.sin(), angles.cos()], dim=-1)
        else:
            temporal = torch.cat([angles.sin(), angles.cos()], dim=-1)[None].expand(
                batch, -1, -1
            )
        feature = self.feature_embedding(torch.arange(channels, device=values.device))
        side = torch.cat(
            [
                temporal[:, :, None, :].expand(-1, -1, channels, -1),
                feature[None, None].expand(batch, length, -1, -1),
                mask.unsqueeze(-1),
            ],
            dim=-1,
        ).permute(0, 3, 2, 1)
        features = torch.stack([values * mask, noisy * (1 - mask)], dim=1).transpose(
            2, 3
        )
        hidden = functional.relu(self.input_projection(features.flatten(2))).reshape(
            batch, self.config.width, channels, length
        )
        embedding = self.step_projection(self.step_table[step])
        skips = []
        for block in self.blocks:
            hidden, skip = block(features=hidden, side=side, step=embedding)
            skips.append(skip)
        output = self.output_projection(
            torch.stack(skips).sum(dim=0).flatten(2) / math.sqrt(len(skips))
        )
        return output.reshape(batch, channels, length).transpose(1, 2) * (1 - mask)

    def training_loss(self, *, batch_x: Tensor, target: Tensor) -> Tensor:
        channels = self.config.input_channels
        initial = batch_x[..., channels : 2 * channels] > 0
        values = torch.where(target.isfinite(), target, batch_x[..., :channels])
        observed = initial | target.isfinite()
        conditional = self.conditional_mask(observed=observed)
        mask = conditional.to(dtype=values.dtype)
        scored = (observed & ~conditional).to(dtype=values.dtype)
        step = torch.randint(
            self.config.diffusion_steps, size=(len(values),), device=values.device
        )
        alpha = self.alpha[step, None, None]
        noise = torch.randn_like(values)
        noisy = alpha.sqrt() * values + (1 - alpha).sqrt() * noise
        error = (
            self.predict_noise(
                noisy=noisy,
                values=values,
                mask=mask,
                step=step,
                times=batch_x[..., 2 * channels]
                if batch_x.shape[-1] > 2 * channels
                else None,
            )
            - noise
        )
        return (error.square() * scored).flatten(1).sum(dim=1).mean()

    def forward(self, *, batch_x: Tensor) -> Tensor:
        generator = torch.Generator(device=batch_x.device).manual_seed(self.config.seed)
        channels = self.config.input_channels
        chunk_size = max(1, MAX_PARALLEL_SAMPLES // len(batch_x))
        samples = []
        for start in range(0, self.config.samples, chunk_size):
            count = min(chunk_size, self.config.samples - start)
            values = batch_x[..., :channels].repeat_interleave(count, dim=0)
            mask = batch_x[..., channels : 2 * channels].repeat_interleave(count, dim=0)
            generated = torch.randn(
                values.shape, device=values.device, generator=generator
            )
            times = (
                batch_x[..., 2 * channels].repeat_interleave(count, dim=0)
                if batch_x.shape[-1] > 2 * channels
                else None
            )
            for index in reversed(range(self.config.diffusion_steps)):
                step = torch.full(
                    (len(values),), index, device=values.device, dtype=torch.long
                )
                noise = self.predict_noise(
                    noisy=generated,
                    values=values,
                    mask=mask,
                    step=step,
                    times=times,
                )
                generated = (
                    generated
                    - self.beta[index] / (1 - self.alpha[index]).sqrt() * noise
                ) / (1 - self.beta[index]).sqrt()
                if index:
                    variance = (
                        self.beta[index]
                        * (1 - self.alpha[index - 1])
                        / (1 - self.alpha[index])
                    )
                    generated = generated + variance.sqrt() * torch.randn(
                        generated.shape, device=values.device, generator=generator
                    )
            generated = values * mask + generated * (1 - mask)
            samples.append(
                generated.reshape(len(batch_x), count, *values.shape[1:]).permute(
                    0, 2, 3, 1
                )
            )
        return torch.cat(samples, dim=-1)
