"""Buffer storage and replay-draw settings shared by the schedules using them."""

from dataclasses import dataclass


class Buffer:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Buffer"
        # Capacity in windows (one window = one input/target pair).
        capacity: int = 256


class Replay:
    @dataclass(kw_only=True)
    class Config(Buffer.Config):
        name: str = "Replay"
        replay: str = "Reservoir"
        replay_size: int = 0
