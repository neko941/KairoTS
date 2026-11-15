"""Time-covariate selection shared by tasks that encode time features."""

from dataclasses import dataclass


class Timed:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Timed"
        # Calendar feature names; their meaning and compatibility belong to the task.
        calendar: tuple[str, ...] = ()
