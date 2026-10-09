"""Integration test of the retrain-and-gate workflow against a real (temporary) MLflow store."""

import pandas as pd
import pytest

pytest.importorskip("mlflow", reason="needs the optional `mlops` extra")

from mlflow import MlflowClient  # noqa: E402

from spam_detector.data import split_dataset  # noqa: E402
from spam_detector.mlops import CHAMPION_ALIAS, MODEL_NAME, retrain  # noqa: E402
from spam_detector.model import MODEL_FILE, load_artifact, load_metadata  # noqa: E402

SPAM = ["free", "win", "prize", "cash", "claim", "urgent", "offer", "reward"]
HAM = ["lunch", "home", "call", "meeting", "tonight", "thanks", "dinner", "later"]


def synthetic_split():
    rows = []
    for i in range(60):
        rows.append((f"{SPAM[i % 8]} {SPAM[(i * 3) % 8]} claim your {SPAM[(i * 5) % 8]} now", 1))
        rows.append((f"{HAM[i % 8]} {HAM[(i * 3) % 8]} see you at {HAM[(i * 5) % 8]} ok", 0))
    df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates("text")
    return split_dataset(df, seed=0)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow writes run artifacts to ./mlruns
    train, test = synthetic_split()
    return {
        "train": train,
        "test": test,
        "kwargs": {
            "params": {"c": 1.0, "analyzer": "word"},
            "seed": 0,
            "threshold": 0.5,
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "artifacts_dir": tmp_path / "artifacts",
            "tags": {"git_sha": "test"},
        },
    }


def run(env, **overrides):
    return retrain(env["train"], env["test"], **{**env["kwargs"], **overrides})


def champion_version() -> str:
    """Registry version holding the `champion` alias (MLflow returns it as an int)."""
    return str(MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version)


def test_first_run_becomes_champion_and_exports_artifacts(env):
    result = run(env)
    assert result["promoted"]
    assert result["serving_version"] == "1"

    assert champion_version() == "1"

    artifacts = env["kwargs"]["artifacts_dir"]
    assert (artifacts / MODEL_FILE).exists()
    assert load_metadata(artifacts)["registry_version"] == "1"
    assert load_artifact(artifacts).predict(["claim your free prize now"])[0] == 1


def test_candidate_that_does_not_improve_is_rejected_but_recorded(env):
    run(env)
    result = run(env)  # identical model: zero gain

    assert not result["promoted"]
    assert result["serving_version"] == "1"

    assert champion_version() == "1"
    rejected = MlflowClient().get_model_version(MODEL_NAME, "2")
    assert rejected.tags["gate"] == "rejected"  # kept in the registry as an audit trail
    assert load_metadata(env["kwargs"]["artifacts_dir"])["registry_version"] == "1"


def test_promotion_moves_the_champion_alias_and_artifacts(env):
    run(env)
    result = run(env, min_improvement=-1.0, max_precision_drop=1.0)  # accept anything

    assert result["promoted"]
    assert champion_version() == "2"
    assert load_metadata(env["kwargs"]["artifacts_dir"])["registry_version"] == "2"
