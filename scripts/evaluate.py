"""Evaluate the saved model on the held-out test split."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from spam_detector.data import DATA_FILE, deduplicate, load_dataset, split_dataset  # noqa: E402
from spam_detector.evaluation import evaluate  # noqa: E402
from spam_detector.model import load_artifact  # noqa: E402

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
SEED = int(os.getenv("RANDOM_SEED", "42"))
THRESHOLD = float(os.getenv("SPAM_THRESHOLD", "0.5"))


def main() -> None:
    df = deduplicate(load_dataset(DATA_DIR / DATA_FILE))
    _, test = split_dataset(df, seed=SEED)

    pipeline = load_artifact(ARTIFACTS_DIR)
    proba = pipeline.predict_proba(test["text"])[:, 1]
    results = evaluate(test["label"], proba, threshold=THRESHOLD)

    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    precision, recall, _ = precision_recall_curve(test["label"], proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title=f"PR curve (AUC = {results['pr_auc']:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "pr_curve.png", dpi=150)

    m = results["at_threshold"]
    print(f"PR-AUC: {results['pr_auc']:.4f}")
    print(
        f"@{m['threshold']}: precision={m['precision']:.4f} "
        f"recall={m['recall']:.4f} f1={m['f1']:.4f}"
    )
    print("\nthreshold  precision  recall   f1     FP   FN")
    for r in results["threshold_sweep"]:
        print(
            f"{r['threshold']:<10} {r['precision']:<10.4f} {r['recall']:<8.4f} "
            f"{r['f1']:<6.4f} {r['false_positives']:<4} {r['false_negatives']}"
        )


if __name__ == "__main__":
    main()
