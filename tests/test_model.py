import json

from spam_detector.model import METADATA_FILE, build_pipeline, load_artifact, save_artifact

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


def fitted_pipeline():
    pipeline = build_pipeline(seed=0)
    pipeline.set_params(tfidf__min_df=1)  # tiny corpus
    return pipeline.fit(TEXTS, LABELS)


def test_pipeline_predicts_probabilities_and_labels():
    pipeline = fitted_pipeline()
    proba = pipeline.predict_proba(["claim your free cash prize now"])[0, 1]
    assert 0.0 <= proba <= 1.0
    assert pipeline.predict(["claim your free cash prize now"])[0] == 1
    assert pipeline.predict(["see you at lunch today"])[0] == 0


def test_artifact_roundtrip(tmp_path):
    pipeline = fitted_pipeline()
    save_artifact(pipeline, {"seed": 0}, tmp_path)

    loaded = load_artifact(tmp_path)
    sample = ["free cash prize", "lunch tonight"]
    assert (loaded.predict_proba(sample) == pipeline.predict_proba(sample)).all()
    assert json.loads((tmp_path / METADATA_FILE).read_text())["seed"] == 0
