"""Train the baseline model and save it with reproducibility metadata."""

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import sklearn

from spam_detector.data import DATA_FILE, deduplicate, load_dataset, split_dataset
from spam_detector.model import build_pipeline, save_artifact

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
SEED = int(os.getenv("RANDOM_SEED", "42"))


def main() -> None:
    data_path = DATA_DIR / DATA_FILE
    df = deduplicate(load_dataset(data_path))
    train, test = split_dataset(df, seed=SEED)

    pipeline = build_pipeline(seed=SEED)
    pipeline.fit(train["text"], train["label"])

    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "n_train": len(train),
        "n_test": len(test),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__},
        "params": {
            "tfidf": {"ngram_range": list(pipeline["tfidf"].ngram_range), "min_df": 2},
            "clf": {
                "class_weight": pipeline["clf"].class_weight,
                "max_iter": pipeline["clf"].max_iter,
            },
        },
    }
    save_artifact(pipeline, metadata, ARTIFACTS_DIR)
    print(f"Saved model to {ARTIFACTS_DIR} (train={len(train)}, test={len(test)})")


if __name__ == "__main__":
    main()
