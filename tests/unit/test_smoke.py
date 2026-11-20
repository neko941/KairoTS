"""Smoke includes pretrained targets and keeps every dependency run temporary."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "smoke", Path(__file__).parents[2] / "scripts/smoke.py"
)
assert _SPEC is not None and _SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


def test_pretrained_smoke_runs_each_source_first_in_the_temporary_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    root = Path(__file__).parents[2]
    targets = [
        *smoke.SOURCES,
        "configs/examples/forecasting/ExchangeRate-ZeroShot.json",
    ]
    paths = {*targets, *smoke.SOURCES.values()}
    paths.add("configs/examples/forecasting/ETDatasetHour-DLinear-Source.json")
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((root / relative).read_text())
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["smoke.py", "--task", "forecasting"])
    calls = []
    runs = {}

    def execute(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        path = Path(arguments[arguments.index("--config") + 1])
        relative = path.relative_to(tmp_path).as_posix()
        store = Path(arguments[arguments.index("--runtime.runs_dir") + 1])
        assert arguments[arguments.index("--runtime.save_checkpoints") + 1] == "true"
        values = json.loads(path.read_text())
        source = values.get("strategy", {}).get("source_config") or smoke.SOURCES.get(
            relative
        )
        if source:
            assert source in calls
            assert (
                arguments[arguments.index("--initialization.run_dir") + 1]
                == runs[source]
            )
        run = store / path.stem
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints/final.pt").touch()
        calls.append(relative)
        runs[relative] = str(run)
        return subprocess.CompletedProcess(
            arguments, 0, json.dumps({"run_dir": str(run)})
        )

    monkeypatch.setattr(smoke.subprocess, "run", execute)
    assert smoke.main() == 0
    assert len(calls) == len(set(calls)) == 6
    assert "5/5 configs ran" in capsys.readouterr().out
    assert all(not Path(run).exists() for run in runs.values())


@pytest.mark.parametrize("cycle", [False, True])
def test_smoke_rejects_missing_or_cyclic_sources_without_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cycle: bool
) -> None:
    directory = tmp_path / "configs/forecasting"
    directory.mkdir(parents=True)
    values = {"initialization": {"name": "Pretrained"}}
    if cycle:
        values["strategy"] = {"source_config": "configs/forecasting/Target.json"}
    (directory / "Target.json").write_text(json.dumps(values))
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["smoke.py"])

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid smoke dependencies must fail before launching")

    monkeypatch.setattr(smoke.subprocess, "run", unexpected)
    assert smoke.main() == 1


@pytest.mark.parametrize("missing_checkpoint", [False, True])
def test_failed_source_prevents_the_pretrained_target_from_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_checkpoint: bool
) -> None:
    for relative, values in (
        ("imputation/Source.json", {}),
        (
            "forecasting/Target.json",
            {"strategy": {"source_config": "configs/imputation/Source.json"}},
        ),
    ):
        path = tmp_path / "configs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values))
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["smoke.py", "--task", "forecasting"])
    calls = []

    def execute(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        path = Path(arguments[arguments.index("--config") + 1])
        calls.append(path.name)
        return subprocess.CompletedProcess(
            arguments,
            0 if missing_checkpoint else 1,
            json.dumps({"run_dir": str(tmp_path / "missing")}),
        )

    monkeypatch.setattr(smoke.subprocess, "run", execute)
    assert smoke.main() == 1
    assert calls == ["Source.json"]
