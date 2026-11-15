"""Settings shared by diffusion methods, with method-specific profile overrides."""

from dataclasses import dataclass


class Diffusion:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Diffusion"
        diffusion_steps: int = 1000
        beta_start: float = 1e-4
        beta_end: float = 0.02


class ConditionalDiffusion:
    @dataclass(kw_only=True)
    class Config(Diffusion.Config):
        name: str = "ConditionalDiffusion"
        samples: int = 100


class FeatureWidth:
    @dataclass(kw_only=True)
    class Config:
        name: str = "FeatureWidth"
        width: int = 64


class ResidualStack:
    @dataclass(kw_only=True)
    class Config(FeatureWidth.Config):
        name: str = "ResidualStack"
        layers: int = 4


class HiddenWidth:
    @dataclass(kw_only=True)
    class Config:
        name: str = "HiddenWidth"
        hidden: int = 64


class Dropout:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Dropout"
        dropout: float = 0.1


class EmbeddingDimensions:
    @dataclass(kw_only=True)
    class Config:
        """Vaswani et al.: d_model embedding width and d_ff intermediate width."""

        name: str = "EmbeddingDimensions"
        d_model: int = 16
        d_ff: int = 32


class EncoderDepth:
    @dataclass(kw_only=True)
    class Config:
        name: str = "EncoderDepth"
        encoder_layers: int = 2
