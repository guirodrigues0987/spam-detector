"""One-command pipeline: download data, train the model, evaluate it.

Usage: python scripts/run_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent
STEPS = ["download_data.py", "train.py", "evaluate.py"]


def main() -> int:
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        result = subprocess.run([sys.executable, str(SCRIPTS / step)], check=False)
        if result.returncode != 0:
            print(f"Step {step} failed", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
