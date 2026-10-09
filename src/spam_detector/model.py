"""Model definition and artifact persistence."""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def build_pipeline(seed: int = 42) -> Pipeline:
    """TF-IDF features + logistic regression, balanced for the rare spam class."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
            ),
        ]
    )


def save_artifact(pipeline: Pipeline, metadata: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / MODEL_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_artifact(directory: Path) -> Pipeline:
    return joblib.load(directory / MODEL_FILE)


def load_metadata(directory: Path) -> dict:
    return json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
