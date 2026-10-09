"""Retrain-and-gate workflow on top of MLflow tracking and the model registry.

Only the scripts and tests import this module: it needs the optional `mlops` extra, which is
deliberately not installed in the production image.
"""

from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from spam_detector.evaluation import evaluate
from spam_detector.model import MODEL_FILE, build_pipeline, save_artifact
from spam_detector.promotion import MAX_PRECISION_DROP, MIN_IMPROVEMENT, decide

MODEL_NAME = "spam-detector"
CHAMPION_ALIAS = "champion"
EXPERIMENT = "spam-detector"


def summarize(y_true, proba, threshold: float) -> dict:
    """The four numbers the gate and the dashboards care about."""
    result = evaluate(y_true, proba, threshold=threshold)
    at = result["at_threshold"]
    return {
        "pr_auc": result["pr_auc"],
        "precision": at["precision"],
        "recall": at["recall"],
        "f1": at["f1"],
    }


def get_champion_version(client: MlflowClient):
    """Registry version currently holding the `champion` alias, or None."""
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None


def retrain(
    train,
    test,
    *,
    params: dict,
    seed: int,
    threshold: float,
    tracking_uri: str,
    artifacts_dir: Path,
    tags: dict | None = None,
    min_improvement: float = MIN_IMPROVEMENT,
    max_precision_drop: float = MAX_PRECISION_DROP,
    run_name: str | None = None,
) -> dict:
    """Train a candidate, track it, apply the quality gate, and export the champion.

    The candidate and the current champion are scored on the *same* test split, so the
    comparison is fair even if the data changed since the champion was trained.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    pipeline = build_pipeline(seed=seed, **params).fit(train["text"], train["label"])
    candidate = summarize(test["label"], pipeline.predict_proba(test["text"])[:, 1], threshold)

    champion_version = get_champion_version(client)
    champion = None
    champion_pipeline = None
    if champion_version is not None:
        champion_pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
        champion = summarize(
            test["label"], champion_pipeline.predict_proba(test["text"])[:, 1], threshold
        )

    decision = decide(candidate, champion, min_improvement, max_precision_drop)
    verdict = "promoted" if decision.promote else "rejected"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**params, "seed": seed, "threshold": threshold})
        mlflow.log_params({"n_train": len(train), "n_test": len(test)})
        mlflow.log_metrics(candidate)
        if champion is not None:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion.items()})
        mlflow.set_tags(
            {
                **(tags or {}),
                "gate": verdict,
                "gate_reason": decision.reason,
            }
        )
        # Every candidate is registered, so the registry keeps an audit trail of what was
        # tried; only the `champion` alias decides what is served.
        info = mlflow.sklearn.log_model(
            pipeline, name="model", registered_model_name=MODEL_NAME
        )
        version = str(info.registered_model_version)
        client.set_model_version_tag(MODEL_NAME, version, "gate", verdict)
        client.set_model_version_tag(MODEL_NAME, version, "gate_reason", decision.reason)

    if decision.promote:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
        serving_pipeline, serving_version, serving_metrics = pipeline, version, candidate
    else:
        serving_pipeline = champion_pipeline
        serving_version = champion_version.version
        serving_metrics = champion

    artifacts_dir = Path(artifacts_dir)
    if decision.promote or not (artifacts_dir / MODEL_FILE).exists():
        # Keep artifacts/ (what the API and the Docker image load) in sync with the champion.
        save_artifact(
            serving_pipeline,
            {
                "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "seed": seed,
                "params": params,
                "metrics": serving_metrics,
                "registry_model": MODEL_NAME,
                "registry_version": str(serving_version),
                "versions": {"scikit-learn": sklearn.__version__, "mlflow": mlflow.__version__},
                **(tags or {}),
            },
            artifacts_dir,
        )

    return {
        "promoted": decision.promote,
        "reason": decision.reason,
        "candidate": candidate,
        "champion": champion,
        "run_id": run.info.run_id,
        "candidate_version": version,
        "serving_version": str(serving_version),
    }
