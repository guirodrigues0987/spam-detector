import json

import pytest

from spam_detector.model import METADATA_FILE, load_artifact, save_artifact
from tests.conftest import fit_tiny_pipeline as fitted_pipeline


def test_vectorizer_is_case_insensitive():
    pipeline = fitted_pipeline()
    lower = pipeline.predict_proba(["claim your free prize"])[0, 1]
    upper = pipeline.predict_proba(["CLAIM YOUR FREE PRIZE"])[0, 1]
    assert lower == pytest.approx(upper)


@pytest.mark.parametrize(
    "text",
    ["", " ", "😀😀😀", "ÀÇÃO não é spam", "12345 67890", "a" * 5000, "free\n\tprize\r\n"],
)
def test_pipeline_handles_unusual_inputs(text):
    proba = fitted_pipeline().predict_proba([text])
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


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
