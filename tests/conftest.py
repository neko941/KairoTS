from __future__ import annotations

import os
import tempfile
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest_plugins = ["pytester"]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
) -> Generator[
    None, Any, None
]:  # positional: pluggy invokes hook arguments positionally.
    outcome = yield
    report = outcome.get_result()
    if (
        os.environ.get("KAIROTS_REQUIRE_REFERENCES") == "1"
        and item.path.name
        in {"test_models.py", "test_partitions.py", "test_runtime.py"}
        and report.skipped
    ):
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid}: required reference check skipped: {report.longrepr}"
        )


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    # Under tmp/ (disposable), never mixed with real runs in runs/.
    scratch = Path(__file__).parents[1] / "tmp" / "tests"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        yield Path(directory)
