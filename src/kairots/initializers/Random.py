"""Random initialization: every parameter keeps its constructor values.

That is PyTorch's seeded random default for most layers; norm layers start at
weight 1, bias 0. DLinear's official `1 / input_length` start uses `Average`
with body-weight patterns.
"""

from dataclasses import dataclass

from kairots.initializers.base import Policy


class Random(Policy):
    @dataclass(kw_only=True)
    class Config(Policy.Config):
        name: str = "Random"
