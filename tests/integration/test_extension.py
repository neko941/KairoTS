"""A strategy's one file supplies discovery, options, requirements and execution."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_one_strategy_file_is_enough(workspace_tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    package = workspace_tmp_path / "src" / "kairots"
    shutil.copytree(
        root / "src" / "kairots", package, ignore=shutil.ignore_patterns("__pycache__")
    )
    (package / "strategies" / "Example.py").write_text(
        "from dataclasses import dataclass\n"
        "from kairots.strategies.FedAvg import FedAvg\n"
        "class Example(FedAvg):\n"
        '    REQUIRES = {"runtime.device": ("cpu",)}\n'
        "    @dataclass(kw_only=True)\n"
        "    class Config(FedAvg.Config):\n"
        '        name: str = "Example"\n'
        "        example_value: int = 7\n"
    )
    script = """
import json
from pathlib import Path
from fixtures.synthetic import ett_experiment, write_ett_sources
from kairots.__main__ import overlay
from kairots.config.schema import Experiment
from kairots.registry import get, names
from kairots.runtime.orchestrator import run
assert 'Example' in names(category='strategies')
assert get(category='strategies',name='Example').Config().example_value == 7
root=Path.cwd()
import kairots
assert Path(kairots.__file__).resolve().is_relative_to(root/'src')
raw=root/'raw'
write_ett_sources(path=raw)
values=ett_experiment(raw_path=raw,runs_dir=root/'runs').to_dict()
values['strategy']={'name':'Example','iterations':1}
config=root/'config.json'
config.write_text(json.dumps(values))
experiment=overlay(path=str(config),arguments=['--strategy.example_value','11'])
assert experiment.strategy.example_value == 11
values['runtime']['device']='auto'
try:
    Experiment.from_dict(values=values)
except ValueError as error:
    assert 'runtime.device' in str(error) and 'Example' in str(error)
else:
    raise AssertionError('unmet requirement accepted')
result=run(experiment=experiment)
assert len(result.rounds)==1
assert result.rounds[0]['selected_clients']
summary=json.loads((Path(result.run_dir)/'summary.json').read_text())
assert summary['status']=='completed'
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=workspace_tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(package.parent), str(root / "tests")]),
        },
        check=True,
    )
