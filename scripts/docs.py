"""Build the HTML docs into docs/_build/html, failing on any warning."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]

if __name__ == "__main__":
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "sphinx", "-W", "--keep-going", "-q", "-b", "html"]
            + [str(ROOT / "docs"), str(ROOT / "docs" / "_build" / "html")]
        )
    )
