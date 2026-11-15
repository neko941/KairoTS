"""Autoregressive multivariate diffusion forecasting with lagged LSTM context."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from kairots.models.base import ConditionalDiffusion, Dropout, ResidualStack
from kairots.models.TimeGrad.Residual import Residual


class TimeGrad(nn.Module):
    """Rasul et al., ICML 2021, arXiv:2101.12072v2."""

    TASKS = ("SampleForecasting",)
    REQUIRES = {
        "learner.name": ("Denoising",),
        "learner.precision": ("fp32",),
    }
    COVARIATES = ("calendar",)

    @dataclass(kw_only=True)
    class Config(ConditionalDiffusion.Config, Dropout.Config, ResidualStack.Config):
        name: str = "TimeGrad"
        input_length: int = 44
        output_length: int = 30
        input_channels: int = 8
        output_channels: int = 8
        calendar_channels: int = 4
        lags: tuple[int, ...] = (1, 7, 14)
        cells: int = 40
        recurrent_layers: int = 2
        conditioning: int = 100
        width: int = 8
        layers: int = 8
        diffusion_steps: int = 100
        beta_end: float = 0.1
        seed: int = 0

        def __post_init__(self) -> None:
            if (
                not self.lags
                or min(self.lags) < 1
                or len(set(self.lags)) != len(self.lags)
            ):
                raise ValueError("TimeGrad needs distinct positive lags")
            if self.input_length <= max(self.lags):
                raise ValueError(
                    "TimeGrad history must include context and maximum lag"
                )
            if self.output_channels != self.input_channels or self.input_channels < 2:
                raise ValueError(
                    "TimeGrad jointly forecasts all channels, at least two"
                )
            if (
                min(
                    self.output_length,
                    self.cells,
                    self.recurrent_layers,
                    self.conditioning,
                    self.width,
                    self.layers,
                    self.diffusion_steps,
                    self.samples,
                )
                < 1
            ):
                raise ValueError("TimeGrad dimensions and counts must be positive")
            if (
                not 0 < self.beta_start <= self.beta_end < 1
                or self.diffusion_steps > 500
            ):
                raise ValueError("TimeGrad needs betas in (0, 1) and at most 500 steps")

    def __init__(self, *, config: Config) -> None:
        super().__init__()
        self.beta: Tensor
        self.alpha: Tensor
        self.step_table: Tensor
        self.config = config
        self.embedding = nn.Embedding(config.input_channels, 1)
        self.recurrent = nn.LSTM(
            config.input_channels * (len(config.lags) + 1) + config.calendar_channels,
            config.cells,
            num_layers=config.recurrent_layers,
            dropout=config.dropout if config.recurrent_layers > 1 else 0,
            batch_first=True,
        )
        self.condition_projection = nn.Linear(config.cells, config.conditioning)
        self.upsampler = nn.Sequential(
            nn.Linear(config.conditioning, config.input_channels // 2),
            nn.LeakyReLU(0.4),
            nn.Linear(config.input_channels // 2, config.input_channels),
            nn.LeakyReLU(0.4),
        )
        angles = torch.arange(1, 501)[:, None] / 10000 ** (
            torch.arange(16)[None, :] * 2 / 32
        )
        self.register_buffer(
            "step_table",
            torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(1),
            persistent=False,
        )
        self.step_projection = nn.Sequential(
            nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.input_projection = nn.Conv1d(1, config.width, kernel_size=1)
        self.blocks = nn.ModuleList(
            Residual(width=config.width, dilation=2 ** (index % 2))
            for index in range(config.layers)
        )
        self.skip_projection = nn.Conv1d(
            config.width,
            config.width,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.output_projection = nn.Conv1d(
            config.width, 1, kernel_size=3, padding=1, padding_mode="circular"
        )
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)
        beta = torch.linspace(
            config.beta_start, config.beta_end, steps=config.diffusion_steps
        )
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", (1 - beta).cumprod(dim=0))

    def context_scale(self, *, history: Tensor) -> Tensor:
        context = self.config.input_length - max(self.config.lags)
        scale = history[:, -context:].mean(dim=1, keepdim=True)
        return torch.where(scale == 0, 1.0, scale)

    def recurrent_features(
        self, *, sequence: Tensor, calendar: Tensor, start: int, end: int
    ) -> Tensor:
        lagged = torch.stack(
            [sequence[:, start - lag : end - lag] for lag in self.config.lags], dim=-1
        ).flatten(2)
        embedded = self.embedding(
            torch.arange(self.config.input_channels, device=sequence.device)
        ).flatten()
        return torch.cat(
            [
                lagged,
                embedded[None, None].expand(len(sequence), end - start, -1),
                calendar[:, start - 1 : end - 1],
            ],
            dim=-1,
        )

    def predict_noise(
        self, *, noisy: Tensor, condition: Tensor, step: Tensor
    ) -> Tensor:
        hidden = functional.relu(self.input_projection(noisy.unsqueeze(1)))
        conditioned = self.upsampler(condition).unsqueeze(1)
        embedding = self.step_projection(self.step_table[step])
        skips = []
        for block in self.blocks:
            hidden, skip = block(features=hidden, condition=conditioned, step=embedding)
            skips.append(skip)
        hidden = torch.stack(skips).sum(dim=0) / math.sqrt(len(skips))
        return self.output_projection(
            functional.relu(self.skip_projection(hidden))
        ).squeeze(1)

    def training_loss(self, *, batch_x: Tensor, target: Tensor) -> Tensor:
        config = self.config
        history = batch_x[:, : config.input_length, : config.input_channels]
        scale = self.context_scale(history=history)
        sequence = torch.cat([history, target], dim=1) / scale
        calendar = batch_x[..., config.input_channels :]
        features = self.recurrent_features(
            sequence=sequence,
            calendar=calendar,
            start=max(config.lags),
            end=config.input_length + config.output_length,
        )
        hidden, _ = self.recurrent(features)
        condition = self.condition_projection(
            hidden[:, -config.output_length :]
        ).flatten(0, 1)
        clean = (target / scale).flatten(0, 1)
        step = torch.randint(
            config.diffusion_steps, size=(len(clean),), device=clean.device
        )
        alpha = self.alpha[step, None]
        noise = torch.randn_like(clean)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        return (
            (self.predict_noise(noisy=noisy, condition=condition, step=step) - noise)
            .square()
            .sum(dim=-1)
            .mean()
        )

    def forward(self, *, batch_x: Tensor) -> Tensor:
        config = self.config
        generator = torch.Generator(device=batch_x.device).manual_seed(config.seed)
        history = batch_x[:, : config.input_length, : config.input_channels]
        scale = self.context_scale(history=history)
        sequence = history / scale
        calendar = batch_x[..., config.input_channels :]
        features = self.recurrent_features(
            sequence=sequence,
            calendar=calendar,
            start=max(config.lags),
            end=config.input_length,
        )
        _, state = self.recurrent(features)
        state = tuple(value.repeat_interleave(config.samples, dim=1) for value in state)
        sequence = sequence.repeat_interleave(config.samples, dim=0)
        calendar = calendar.repeat_interleave(config.samples, dim=0)
        generated_steps = []
        for position in range(
            config.input_length, config.input_length + config.output_length
        ):
            features = self.recurrent_features(
                sequence=sequence, calendar=calendar, start=position, end=position + 1
            )
            hidden, state = self.recurrent(features, state)
            condition = self.condition_projection(hidden[:, 0])
            generated = torch.randn(
                (len(sequence), config.input_channels),
                device=batch_x.device,
                generator=generator,
            )
            for index in reversed(range(config.diffusion_steps)):
                step = torch.full(
                    (len(sequence),), index, device=batch_x.device, dtype=torch.long
                )
                noise = self.predict_noise(
                    noisy=generated, condition=condition, step=step
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
                        generated.shape, device=batch_x.device, generator=generator
                    )
            generated_steps.append(generated)
            sequence = torch.cat([sequence, generated[:, None]], dim=1)
        samples = torch.stack(generated_steps, dim=1) * scale.repeat_interleave(
            config.samples, dim=0
        )
        return samples.reshape(
            len(batch_x), config.samples, config.output_length, config.input_channels
        ).permute(0, 2, 3, 1)
