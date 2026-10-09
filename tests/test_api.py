import logging

import pytest

from tests.conftest import HAM_TEXT, SPAM_TEXT, make_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "test-version",
    }


def test_predict_spam(client):
    response = client.post("/predict", json={"text": SPAM_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "spam"
    assert 0.5 <= body["spam_probability"] <= 1.0
    assert body["threshold"] == 0.5
    assert body["model_version"] == "test-version"


def test_predict_ham(client):
    body = client.post("/predict", json={"text": HAM_TEXT}).json()
    assert body["label"] == "ham"
    assert body["spam_probability"] < 0.5


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "   "},
        {"text": "x" * 1001},
        {},
        {"text": 123},
        {"text": None},
        {"message": "wrong field name"},
    ],
)
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_non_json_body(client):
    response = client.post("/predict", content="not json", headers={"content-type": "text/plain"})
    assert response.status_code == 422


def test_predict_accepts_text_at_max_length(client):
    assert client.post("/predict", json={"text": "a" * 1000}).status_code == 200


@pytest.mark.parametrize(("threshold", "expected"), [(0.0, "spam"), (1.01, "ham")])
def test_threshold_controls_label(model_dir, threshold, expected):
    with make_client(model_dir, threshold=threshold) as c:
        body = c.post("/predict", json={"text": HAM_TEXT}).json()
    assert body["label"] == expected
    assert body["threshold"] == threshold


def test_service_is_unavailable_without_model(tmp_path):
    with make_client(tmp_path) as c:  # empty dir: no model to load
        health = c.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "unavailable"
        assert c.post("/predict", json={"text": SPAM_TEXT}).status_code == 503


def test_request_id_is_generated_and_echoed(client):
    generated = client.get("/health").headers["x-request-id"]
    assert generated
    echoed = client.get("/health", headers={"x-request-id": "abc-123"}).headers["x-request-id"]
    assert echoed == "abc-123"


def test_message_text_is_never_logged(client, caplog):
    secret = "my secret code is 998877 please claim your free prize"
    with caplog.at_level(logging.INFO, logger="spam_detector.api"):
        client.post("/predict", json={"text": secret})
    logged = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert "prediction" in logged
    assert "998877" not in logged
    assert any(r.__dict__.get("text_length") == len(secret) for r in caplog.records)
