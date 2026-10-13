import re
import urllib.request

from prometheus_client import generate_latest

from tests.conftest import HAM_TEXT, SPAM_TEXT


def sample(body: str, name: str, **labels) -> float:
    """Read one sample value from Prometheus exposition text."""
    wanted = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    pattern = rf"^{name}\{{{re.escape(wanted)}\}}\s+(\S+)$" if labels else rf"^{name}\s+(\S+)$"
    match = re.search(pattern, body, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def scrape(client) -> str:
    """Read the app's registry directly; the HTTP server has its own test below."""
    return generate_latest(client.app.state.metrics.registry).decode()


def test_metrics_are_not_served_on_the_public_api_port(client):
    assert client.get("/metrics").status_code == 404
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]


def test_no_metrics_server_runs_unless_a_port_is_configured(client):
    assert client.app.state.metrics_server is None


def test_metrics_are_served_over_http_on_the_configured_port(model_dir):
    from tests.conftest import make_client

    with make_client(model_dir, metrics_port=0) as client:  # port 0: the OS picks a free one
        client.post("/predict", json={"text": SPAM_TEXT})
        port = client.app.state.metrics_server.server_port
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/plain")
            body = response.read().decode()
    assert sample(body, "spam_predictions_total", label="spam") == 1


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


def test_model_version_is_exposed(client):
    assert sample(scrape(client), "spam_model_info", version="test-version") == 1


def test_each_app_has_an_independent_registry(model_dir):
    from tests.conftest import make_client

    with make_client(model_dir) as first, make_client(model_dir) as second:
        first.post("/predict", json={"text": SPAM_TEXT})
        assert sample(scrape(first), "spam_predictions_total", label="spam") == 1
        assert sample(scrape(second), "spam_predictions_total", label="spam") == 0
