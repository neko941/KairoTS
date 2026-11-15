"""Forgetting, plasticity and final performance from a task-wise error matrix.

`matrix[i][j]` is the error of the model after task `i` on task `j`'s test windows
(lower is better), defined for `j <= i`, as in Hallak and Kem (arXiv 2510.21491).
"""


def forecasting_summary(*, matrix: list[list[float]]) -> dict[str, float]:
    """AF = mean_{j<N}(P[N,j] - P[j,j]); AP = mean_j P[j,j]; AvgPerf = mean_j P[N,j]."""
    last = matrix[-1]
    diagonal = [row[task] for task, row in enumerate(matrix)]
    return {
        "AF": sum(last[task] - diagonal[task] for task in range(len(last) - 1))
        / (len(last) - 1),
        "AP": sum(diagonal) / len(diagonal),
        "AvgPerf": sum(last) / len(last),
    }
