import re

from tests.conftest import HAM_TEXT, SPAM_TEXT


def sample(body: str, name: str, **labels) -> float:
    """Read one sample value from Prometheus exposition text."""
    wanted = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    pattern = rf"^{name}\{{{re.escape(wanted)}\}}\s+(\S+)$" if labels else rf"^{name}\s+(\S+)$"
    match = re.search(pattern, body, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def scrape(client) -> str:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return response.text


def test_metrics_endpoint_is_hidden_from_the_api_docs(client):
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]


def test_predictions_are_counted_by_label(client):
    client.post("/predict", json={"text": SPAM_TEXT})
    client.post("/predict", json={"text": HAM_TEXT})
    client.post("/predict", json={"text": HAM_TEXT})

    body = scrape(client)
    assert sample(body, "spam_predictions_total", label="spam") == 1
    assert sample(body, "spam_predictions_total", label="ham") == 2
    assert sample(body, "spam_probability_count") == 3


def test_requests_are_counted_by_route_and_status(client):
    client.post("/predict", json={"text": SPAM_TEXT})
    client.post("/predict", json={"text": ""})  # 422

    body = scrape(client)
    ok = sample(body, "http_requests_total", method="POST", path="/predict", status="200")
    bad = sample(body, "http_requests_total", method="POST", path="/predict", status="422")
    assert (ok, bad) == (1, 1)
    assert sample(body, "http_request_duration_seconds_count", method="POST", path="/predict") == 2


def test_unknown_urls_share_one_label_to_bound_cardinality(client):
    for i in range(5):
        client.get(f"/does-not-exist/{i}")

    body = scrape(client)
    assert sample(body, "http_requests_total", method="GET", path="unmatched", status="404") == 5
    assert "does-not-exist" not in body


def test_scraping_metrics_does_not_count_itself(client):
    scrape(client)
    assert 'path="/metrics"' not in scrape(client)


def test_model_version_is_exposed(client):
    assert sample(scrape(client), "spam_model_info", version="test-version") == 1


def test_each_app_has_an_independent_registry(model_dir):
    from tests.conftest import make_client

    with make_client(model_dir) as first, make_client(model_dir) as second:
        first.post("/predict", json={"text": SPAM_TEXT})
        assert sample(scrape(first), "spam_predictions_total", label="spam") == 1
        assert sample(scrape(second), "spam_predictions_total", label="spam") == 0
