"""Initialization policy contract: ordered rules from parameter-name patterns to
named initializers, applied once to the canonical global model after construction."""

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.init import constant_, normal_, ones_, xavier_uniform_, zeros_


def inverse_fan_in_(tensor: Tensor) -> Tensor:  # positional: torch.nn.init signature.
    """Every weight `1 / fan_in`: a linear layer starts as the mean of its inputs."""
    return constant_(tensor, val=1 / tensor.shape[1])


INITIALIZERS = {
    "inverse_fan_in": inverse_fan_in_,
    "normal": normal_,
    "ones": ones_,
    "xavier_uniform": xavier_uniform_,
    "zeros": zeros_,
}


class Policy:
    ABSTRACT = True
    # (parameter-name pattern, INITIALIZERS name, keyword options); unmatched
    # parameters keep the recipe's native initialization.
    RULES: tuple[tuple[str, str, dict[str, Any]], ...] = ()

    @dataclass(kw_only=True)
    class Config:
        name: str = ""

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def apply(
        self, *, model: nn.Module, manifest: dict[str, Any] | None = None
    ) -> dict[str, str]:
        """Initialize matched parameters in place; return parameter -> initializer.

        A rule that matches nothing, or a parameter two rules match, is an error.
        Tied parameters appear once in `named_parameters`, so each is set once.
        `manifest` is this run's saved manifest, for policies that check it.
        """
        parameters = dict(model.named_parameters())
        applied: dict[str, str] = {}
        for pattern, initializer, options in self.RULES:
            matched = [name for name in parameters if fnmatch(name, pattern)]
            if not matched:
                raise ValueError(
                    f"{type(self).__name__}: rule {pattern!r} matches no parameter"
                )
            for name in matched:
                if name in applied:
                    raise ValueError(
                        f"{type(self).__name__}: {name!r} matches two rules"
                    )
                with torch.no_grad():
                    INITIALIZERS[initializer](parameters[name], **options)
                applied[name] = initializer
        return applied
