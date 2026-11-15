"""BasicMotions: smart-watch accelerometer and gyroscope, 4 activities.

40 train and 40 test samples, 6 channels, length 100, sampled at 10 Hz.
"""

from dataclasses import dataclass

from kairots.data.sources.UEA import UEA


class BasicMotions(UEA):
    TASKS = ("Classification",)
    PROBLEM = "BasicMotions"
    SHA256 = {
        "TRAIN": "8dc43cc6306cb679c888c01e26f91772ac4441a916da43bac8b79734a538b9d6",
        "TEST": "79213102bc6fca1a398ad98ce1185dff0208fa3d1465e687f48288946b0ff8dc",
    }

    @dataclass(kw_only=True)
    class Config(UEA.Config):
        name: str = "BasicMotions"
        input_channels: tuple[str, ...] = (
            "accelerometer_x",
            "accelerometer_y",
            "accelerometer_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        )
        input_length: int = 100
        num_classes: int = 4
