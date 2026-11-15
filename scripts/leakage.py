"""Leakage check: training must not depend on the test set.

Usage: python scripts/leakage.py
Each check trains twice, once with every test value (or test sample) replaced, and
fails unless every train/validation number and the trained parameters are bitwise
identical while the test scores change. It covers series forecasting (FedAvg,
Ditto, Prequential, TAFAS), classification (Frozen, TENT) and M4's official test
fold, on synthetic data (tests/integration/test_leakage.py). It also checks that
TAFAS's predictions and updates never read a target the stream has not yet observed,
that the default anomaly threshold is computed without test values and that horizon
values of forecast targets and past-observed covariates never reach a model.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def main() -> int:
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/integration/test_leakage.py",
            "tests/integration/test_tafas.py::test_unobserved_targets_never_reach_an_earlier_prediction",
            "tests/integration/test_continual.py::test_future_round_values_do_not_change_earlier_training",
            "tests/integration/test_groups.py::test_group_holdouts_never_train",
            "tests/integration/test_anomaly_detection.py::test_default_threshold_uses_no_test_data",
            "tests/unit/test_covariates.py::test_future_values_of_targets_and_past_covariates_never_reach_the_model",
            "tests/unit/test_covariates.py::test_tide_reads_no_unavailable_value",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
