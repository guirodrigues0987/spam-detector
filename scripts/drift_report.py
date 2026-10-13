"""Compare what the API received against the training data and write an Evidently report.

    docker compose logs api --no-log-prefix > logs/api.log
    python scripts/drift_report.py --logs logs/api.log

Needs the optional `monitoring` extra. The model in ARTIFACTS_DIR must be the one that served
the logs, because the predicted-probability column is compared against its scores on the
reference data.
"""

import argparse
import os
import sys
from pathlib import Path

from spam_detector.data import DATA_FILE, deduplicate, load_dataset, split_dataset
from spam_detector.model import load_artifact
from spam_detector.monitoring import (
    build_drift_report,
    parse_prediction_logs,
    read_log_lines,
    reference_frame,
    summarize_drift,
)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
SEED = int(os.getenv("RANDOM_SEED", "42"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, required=True, help="file with the API's JSON logs")
    parser.add_argument("--out", type=Path, default=Path("reports/drift.html"))
    parser.add_argument("--reference-size", type=int, default=2000)
    parser.add_argument("--min-rows", type=int, default=50, help="refuse to compare tiny samples")
    parser.add_argument("--fail-on-drift", action="store_true", help="exit 2 if drift is found")
    args = parser.parse_args()

    current = parse_prediction_logs(read_log_lines(args.logs))
    if len(current) < args.min_rows:
        print(
            f"Only {len(current)} predictions in the logs (need {args.min_rows}).",
            file=sys.stderr,
        )
        return 1

    train, _ = split_dataset(deduplicate(load_dataset(DATA_DIR / DATA_FILE)), seed=SEED)
    sample = train.sample(n=min(args.reference_size, len(train)), random_state=SEED)
    model = load_artifact(ARTIFACTS_DIR)
    reference = reference_frame(sample["text"], model.predict_proba(sample["text"])[:, 1])

    snapshot = build_drift_report(reference, current)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.out))
    summary = summarize_drift(snapshot.dict())

    print(f"reference: {len(reference)} rows | current: {len(current)} rows")
    for name, info in summary["columns"].items():
        flag = "DRIFT" if info["drifted"] else "ok"
        print(f"  {name:<16} {flag:<6} {info['method']} = {info['value']:.4g}")
    print(f"drifted columns: {summary['drifted_share']:.0%}")
    print(f"report: {args.out}")
    return 2 if args.fail_on_drift and summary["drifted_share"] >= 0.5 else 0


if __name__ == "__main__":
    sys.exit(main())
