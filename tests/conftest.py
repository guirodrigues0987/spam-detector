"""Shared fixtures. A tiny model is trained on the fly so tests need no downloaded data."""

import pytest
from fastapi.testclient import TestClient

from spam_detector.api import create_app
from spam_detector.config import Settings
from spam_detector.model import build_pipeline, save_artifact

TEXTS = [
    "win a free prize now, claim your cash",
    "free entry, text WIN to 80000 now",
    "urgent: claim your free prize today",
    "congratulations, you won free cash now",
    "are we still meeting for lunch today",
    "see you at home later tonight",
    "can you call me when you get back",
    "lunch today at noon, see you there",
]
LABELS = [1, 1, 1, 1, 0, 0, 0, 0]

SPAM_TEXT = "claim your free cash prize now"
HAM_TEXT = "see you at lunch today"


def fit_tiny_pipeline():
    pipeline = build_pipeline(seed=0)
    pipeline.set_params(tfidf__min_df=1)  # tiny corpus
    return pipeline.fit(TEXTS, LABELS)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("artifacts")
    save_artifact(fit_tiny_pipeline(), {"trained_at": "test-version"}, path)
    return path


def make_client(artifacts_dir, threshold: float = 0.5) -> TestClient:
    settings = Settings(artifacts_dir=artifacts_dir, spam_threshold=threshold, log_level="INFO")
    return TestClient(create_app(settings))


@pytest.fixture
def client(model_dir):
    with make_client(model_dir) as c:  # context manager runs the lifespan (loads the model)
        yield c
