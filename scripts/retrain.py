"""Train a candidate model, track it in MLflow and promote it only if it beats the champion.

Examples:
    python scripts/retrain.py                          # baseline (first run becomes champion)
    python scripts/retrain.py --analyzer char_wb       # challenger with character n-grams
    python scripts/retrain.py --c 0.01                 # a deliberately weaker challenger

Browse the runs with: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

from spam_detector.data import DATA_FILE, deduplicate, load_dataset, split_dataset
from spam_detector.mlops import retrain
from spam_detector.promotion import MAX_PRECISION_DROP, MIN_IMPROVEMENT

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
SEED = int(os.getenv("RANDOM_SEED", "42"))
THRESHOLD = float(os.getenv("SPAM_THRESHOLD", "0.5"))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--c", type=float, default=1.0, help="inverse regularization strength")
    parser.add_argument("--analyzer", choices=["word", "char_wb"], default="word")
    parser.add_argument("--min-improvement", type=float, default=MIN_IMPROVEMENT)
    parser.add_argument("--max-precision-drop", type=float, default=MAX_PRECISION_DROP)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    data_path = DATA_DIR / DATA_FILE
    train, test = split_dataset(deduplicate(load_dataset(data_path)), seed=SEED)

    result = retrain(
        train,
        test,
        params={"c": args.c, "analyzer": args.analyzer},
        seed=SEED,
        threshold=THRESHOLD,
        tracking_uri=TRACKING_URI,
        artifacts_dir=ARTIFACTS_DIR,
        tags={
            "git_sha": git_sha(),
            "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        },
        min_improvement=args.min_improvement,
        max_precision_drop=args.max_precision_drop,
        run_name=args.run_name,
    )

    def fmt(m):
        return "n/a" if m is None else f"PR-AUC={m['pr_auc']:.4f} precision={m['precision']:.4f}"

    print(f"candidate (v{result['candidate_version']}): {fmt(result['candidate'])}")
    print(f"champion:  {fmt(result['champion'])}")
    print(f"decision:  {'PROMOTED' if result['promoted'] else 'REJECTED'} - {result['reason']}")
    print(f"serving:   registry version {result['serving_version']} (exported to {ARTIFACTS_DIR})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
