"""Quick test: run every config under configs/ for one iteration on the CPU.

Usage: python scripts/smoke.py [--task forecasting]
Raw data downloads on first use; runs go to a temporary directory and are deleted.
Pretrained examples first run their source config and use its temporary checkpoint.
Sampling models draw one sample, so the smoke checks the path, not the sample count.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCES = {
    "configs/forecasting/ETDatasetHour-TimesNet-FineTune.json": (
        "configs/imputation/ETDatasetHour-TimesNet-Pretrain.json"
    ),
    "configs/forecasting/ETTh2-TimesNet-Transfer.json": (
        "configs/forecasting/ETTh1-TimesNet-Source.json"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="only configs/<task>/")
    task = parser.parse_args().task
    configs = sorted((ROOT / "configs").glob(f"{task or '*'}/*.json"))
    failed: list[str] = []
    with tempfile.TemporaryDirectory() as runs_dir:
        completed: dict[Path, str] = {}
        active: set[Path] = set()

        def execute(*, config: Path) -> str:
            config = config.resolve()
            if config in completed:
                return completed[config]
            if config in active:
                raise ValueError(f"cyclic smoke source: {config}")
            active.add(config)
            values = json.loads(config.read_text(encoding="utf-8"))
            name = config.relative_to(ROOT.resolve()).as_posix()
            source = values.get("strategy", {}).get("source_config") or SOURCES.get(
                name
            )
            overrides = []
            if source:
                overrides = [
                    "--initialization.run_dir",
                    execute(config=ROOT / source),
                ]
            elif values.get("initialization", {}).get("name") == "Pretrained":
                raise ValueError(f"no chained smoke source for {name}")
            if "samples" in values.get("model", {}):
                overrides += ["--model.samples", "1"]
            print(f"smoke {name}", flush=True)
            process = subprocess.run(
                [sys.executable, "-m", "kairots", "--config", str(config)]
                + ["--strategy.iterations", "1", "--runtime.device", "cpu"]
                + ["--runtime.runs_dir", runs_dir]
                + ["--runtime.save_checkpoints", "true"]
                + overrides,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                text=True,
            )
            if process.returncode:
                raise RuntimeError(f"smoke failed: {name}")
            run_dir = json.loads(process.stdout)["run_dir"]
            checkpoint = ROOT / run_dir / "checkpoints" / "final.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"smoke checkpoint missing: {checkpoint}")
            completed[config] = run_dir
            active.remove(config)
            return run_dir

        for config in configs:
            name = config.relative_to(ROOT).as_posix()
            try:
                execute(config=config)
            except (OSError, ValueError, RuntimeError) as error:
                print(error, file=sys.stderr)
                failed.append(name)
                active.clear()
    print(f"{len(configs) - len(failed)}/{len(configs)} configs ran")
    for name in failed:
        print(f"FAILED {name}")
    return 1 if failed or not configs else 0


if __name__ == "__main__":
    sys.exit(main())
