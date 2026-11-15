"""Format KairoTS code, then check the style rules that cannot be autofixed.

Fix mode: ruff autofix (unused imports, import order), then the ruff formatter.
Both modes then check: ruff lint (type hints ANN, names N, unused variables F841)
and scripts/check_style.py (keyword-only defs, annotations, glossary names).
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
TARGETS = ["src", "tests", "scripts"]


def run(*, args: list[str]) -> int:
    print("$ " + " ".join(args), flush=True)
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report only; write nothing"
    )
    check = parser.parse_args().check
    codes = []
    if check:
        codes.append(run(args=["-m", "ruff", "format", "--check", *TARGETS]))
    else:
        codes.append(run(args=["-m", "ruff", "check", "--fix", "-q", *TARGETS]))
        codes.append(run(args=["-m", "ruff", "format", *TARGETS]))
    codes.append(run(args=["-m", "ruff", "check", *TARGETS]))
    codes.append(run(args=["scripts/check_style.py"]))
    return max(codes)


if __name__ == "__main__":
    sys.exit(main())
