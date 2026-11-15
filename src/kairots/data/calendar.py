"""Known-future calendar features of each step, as TSLib's `timeF` encoding.

Each feature maps a timestamp to [-0.5, 0.5] (`utils/timefeatures.py`). A
timestamp is known for every step of the horizon, so these features may be given
to a model for the forecast window as well as the history.
"""

import math

import polars as pl
import torch

FEATURES = {
    "ElapsedHours": (
        pl.col("date").dt.epoch(time_unit="ms")
        - pl.col("date").dt.epoch(time_unit="ms").first()
    )
    / 3_600_000,
    "MinuteOfHour": pl.col("date").dt.minute() / 59.0 - 0.5,
    "HourOfDay": pl.col("date").dt.hour() / 23.0 - 0.5,
    "DayOfWeek": (pl.col("date").dt.weekday() - 1) / 6.0 - 0.5,
    "DayOfMonth": (pl.col("date").dt.day() - 1) / 30.0 - 0.5,
    "DayOfYear": (pl.col("date").dt.ordinal_day() - 1) / 365.0 - 0.5,
    "MonthOfYear": (pl.col("date").dt.month() - 1) / 11.0 - 0.5,
}

# Fourier periods match the author covariates on a complete daily/hourly grid.
for _name, _value, _period in (
    ("Weekday", pl.col("date").dt.weekday() - 1, 7),
    ("Hour", pl.col("date").dt.hour(), 24),
):
    FEATURES[f"{_name}Cos"] = (_value * (2 * math.pi / _period)).cos()
    FEATURES[f"{_name}Sin"] = (_value * (2 * math.pi / _period)).sin()


def features(*, dates: pl.Series, names: tuple[str, ...]) -> torch.Tensor:
    """[steps, len(names)] calendar features of `dates`, in `names` order."""
    year_fourier = {"DayOfYearCos", "DayOfYearSin"}
    unknown = set(names) - FEATURES.keys() - year_fourier
    if unknown:
        raise ValueError(f"unknown calendar features {sorted(unknown)}")
    if not dates.dtype.is_temporal():
        raise ValueError("calendar features need datetime dates")
    year_period = 0
    if set(names) & year_fourier:
        ordinal_day = dates.dt.ordinal_day().max()
        if not isinstance(ordinal_day, int):
            raise ValueError("day-of-year Fourier features need at least one date")
        year_period = ordinal_day + 1
    expressions = []
    for name in names:
        if name in year_fourier:
            ordinal = pl.col("date").dt.ordinal_day()
            angle = ordinal * (2 * math.pi / year_period)
            expressions.append(
                (angle.cos() if name.endswith("Cos") else angle.sin()).alias(name)
            )
        else:
            expressions.append(FEATURES[name].alias(name))
    frame = pl.DataFrame({"date": dates}).select(expressions)
    return torch.from_numpy(frame.to_numpy().astype("float32"))
