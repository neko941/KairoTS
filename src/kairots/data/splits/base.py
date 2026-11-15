"""Narrow schemas for window geometry, eligibility and training-window strides."""

from dataclasses import dataclass


class Minimum:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Minimum"
        min_windows: int = 10


class Offset:
    @dataclass(kw_only=True)
    class Config:
        offset_length: int = 0


class Window:
    @dataclass(kw_only=True)
    class Config(Minimum.Config, Offset.Config):
        name: str = "Window"
        input_length: int = 96
        output_length: int = 24


class Strided:
    @dataclass(kw_only=True)
    class Config(Minimum.Config, Offset.Config):
        name: str = "Strided"
        train_stride: int = 1


class Segments:
    """Ordered time tasks, specified by lengths or explicit half-open row ranges."""

    @dataclass(kw_only=True)
    class Config:
        name: str = "Segments"
        base_length: int = 8760
        task_length: int = 2184
        task_ranges: tuple[tuple[int, int], ...] = ()

        def __post_init__(self) -> None:
            if self.base_length < 1 or self.task_length < 1:
                raise ValueError("segments need positive lengths")
            previous = 0
            for span in self.task_ranges:
                if (
                    not isinstance(span, (list, tuple))
                    or len(span) != 2
                    or any(type(value) is not int for value in span)
                ):
                    raise ValueError("task_ranges must contain integer row pairs")
                first, last = span
                if first < previous or last <= first:
                    raise ValueError(
                        "task_ranges must be ordered, nonoverlapping ranges"
                    )
                previous = last
            self.task_ranges = tuple((span[0], span[1]) for span in self.task_ranges)
