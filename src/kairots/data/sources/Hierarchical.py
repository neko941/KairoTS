"""Structure kind for series that aggregate along a hierarchy."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from kairots.data.sources.Multivariate import Multivariate


@dataclass(kw_only=True)
class Hierarchical(Multivariate):
    """Every series of the hierarchy is a channel of one frame. `manifest_fields`
    carries `bottom`, the bottom-level channels, and `summing`, the summing matrix
    as each channel's bottom-level members (a bottom channel lists itself).
    """

    ABSTRACT = True

    @staticmethod
    def summing(*, frame: pl.DataFrame, matrix: pl.DataFrame) -> dict[str, list[str]]:
        """Read a summing matrix (one row per series, one 0/1 column per bottom
        series, the first column naming the row) and check that `frame` adds up.
        """
        names = matrix.get_column(matrix.columns[0]).to_list()
        bottom = matrix.columns[1:]
        values = matrix.select(bottom).to_numpy()
        totals = frame.select(bottom).to_numpy() @ values.T
        if not np.allclose(totals, frame.select(names).to_numpy(), rtol=1e-6):
            raise ValueError("series do not equal the sums of their bottom members")
        return {
            name: [series for series, flag in zip(bottom, row, strict=True) if flag]
            for name, row in zip(names, values, strict=True)
        }
