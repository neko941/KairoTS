"""Split complete forecasting windows with disjoint target periods."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Literal

import numpy as np
import polars as pl

from kairots.data.splits.base import Window


@dataclass(kw_only=True)
class CompleteWindowRatio:
    config: CompleteWindowRatio.Config
    """Create ordered target periods from valid windows."""

    @dataclass(kw_only=True)
    class Config(Window.Config):
        name: str = "CompleteWindowRatio"
        # Train, validation and test shares of each client's rows; validation is opt-in
        # (e.g. 0.7, 0.1, 0.2).
        ratios: tuple[float, float, float] = (0.8, 0.0, 0.2)
        cut: Literal["per_client", "shared_calendar"] = "per_client"
        lookback_crosses_boundary: bool = True
        # Few-shot budget: keep each client's last train_windows train windows (the
        # most recent before the cut); 0 keeps all.
        train_windows: int = 0
        stride: int = 1
        # Rows where the validation and test target periods begin and where the used
        # rows end, replacing `ratios` (the fixed borders of the ETT benchmark
        # loaders, which ignore rows after the last border); all 0 is off.
        cut_rows: tuple[int, int, int] = (0, 0, 0)

        def __post_init__(self) -> None:
            if self.stride < 1:
                raise ValueError("window stride must be positive")
            first, second, last = self.cut_rows
            if any(self.cut_rows) and not 0 < first < second < last:
                raise ValueError("cut_rows must be increasing positive row positions")

    def __init__(self, *, config: Config | None = None) -> None:
        self.config = config or self.Config()

    @staticmethod
    def _grid(*, dates: list[Any]) -> list[Any]:
        """Positions for the continuity check: month starts count in months, since
        calendar months differ in length."""
        month_starts = all(
            isinstance(value, date)
            and value.day == 1
            and (not isinstance(value, datetime) or value.time() == time())
            for value in dates
        )
        return (
            [value.year * 12 + value.month for value in dates]
            if month_starts
            else dates
        )

    @classmethod
    def _starts(
        cls, *, frame: pl.DataFrame, channels: tuple[str, ...], length: int
    ) -> np.ndarray:
        """Starts of the `length`-row windows with finite values on an unbroken
        time grid (the grid's smallest step), from prefix counts of bad rows and
        bad gaps."""
        count = len(frame) - length + 1
        if count <= 0:
            return np.zeros(0, dtype=np.int64)
        values = frame.select(channels).to_numpy()
        bad_rows = np.concatenate(
            ([0], np.cumsum(~np.isfinite(values).all(axis=1), dtype=np.int64))
        )
        starts = np.arange(count)
        complete = bad_rows[starts + length] == bad_rows[starts]
        grid = cls._grid(dates=frame.get_column("date").to_list())
        if length >= 2 and len(grid) >= 2:
            differences = [
                grid[index + 1] - grid[index] for index in range(len(grid) - 1)
            ]
            step = min(differences)
            bad_gaps = np.concatenate(
                (
                    [0],
                    np.cumsum(
                        [difference != step for difference in differences],
                        dtype=np.int64,
                    ),
                )
            )
            complete &= bad_gaps[starts + length - 1] == bad_gaps[starts]
        return starts[complete]

    def shared_cut_dates(
        self,
        *,
        frames: list[pl.DataFrame],
        input_length: int,
        output_length: int,
        offset_length: int,
        ratios: tuple[float, float, float],
        channels: tuple[str, ...],
    ) -> tuple[object, object]:
        timestamps: list[object] = []
        for frame in frames:
            dates = frame.get_column("date").to_list()
            timestamps += [
                dates[start + input_length + offset_length]
                for start in self._starts(
                    frame=frame,
                    channels=channels,
                    length=input_length + offset_length + output_length,
                ).tolist()
            ]
        timestamps.sort(
            key=lambda value: (
                value.isoformat()
                if isinstance(value, (date, datetime))
                else f"{value:020}"
            )
        )
        if len(timestamps) < 3:
            raise ValueError("not enough complete windows to find shared calendar cuts")
        first = min(len(timestamps) - 1, int(len(timestamps) * ratios[0]))
        second = min(
            len(timestamps) - 1, int(len(timestamps) * (ratios[0] + ratios[1]))
        )
        return timestamps[first], timestamps[second]

    def split(
        self,
        *,
        frame: pl.DataFrame,
        input_length: int,
        output_length: int,
        offset_length: int,
        ratios: tuple[float, float, float],
        channels: tuple[str, ...],
        lookback_crosses_boundary: bool,
        min_windows: int,
        cut_dates: tuple[object, object] | None = None,
    ) -> dict[str, list[tuple[int, int]]]:
        dates = frame.get_column("date").to_list()
        complete = [
            (
                start,
                start + input_length + offset_length,
                start + input_length + offset_length + output_length,
            )
            for start in self._starts(
                frame=frame,
                channels=channels,
                length=input_length + offset_length + output_length,
            ).tolist()
        ]
        if any(self.config.cut_rows):
            complete = [
                window for window in complete if window[2] <= self.config.cut_rows[2]
            ]
        # The store checks each enabled split against min_windows.
        if not complete:
            return {"train": [], "validation": [], "test": []}
        if any(self.config.cut_rows):
            first_cut, second_cut, _ = self.config.cut_rows
        elif cut_dates is None:
            # Ratios of rows, not of windows, so the borders do not move with the
            # input, offset or output length.
            first_cut = int(len(frame) * ratios[0])
            second_cut = int(len(frame) * (ratios[0] + ratios[1]))
        else:
            first_cut = bisect_left(dates, cut_dates[0])
            second_cut = bisect_left(dates, cut_dates[1])
        assignments = {
            "train": [
                (start, target)
                for start, target, end in complete
                if target < first_cut and end <= first_cut
            ],
            "validation": [
                (start, target)
                for start, target, end in complete
                if first_cut <= target < second_cut
                and end <= second_cut
                and (lookback_crosses_boundary or start >= first_cut)
            ],
            "test": [
                (start, target)
                for start, target, end in complete
                if target >= second_cut
                and (lookback_crosses_boundary or start >= second_cut)
            ],
        }
        return {
            name: windows[:: self.config.stride]
            for name, windows in assignments.items()
        }
