"""Prometheus metrics. Each app gets its own registry, so tests can create many apps."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
PROBABILITY_BUCKETS = tuple(round(i / 10, 1) for i in range(1, 10)) + (1.0,)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds",
            ["method", "path"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "spam_predictions_total",
            "Predictions by predicted class",
            ["label"],
            registry=self.registry,
        )
        self.probability = Histogram(
            "spam_probability",
            "Distribution of predicted spam probabilities",
            buckets=PROBABILITY_BUCKETS,
            registry=self.registry,
        )
        self.model_info = Gauge(
            "spam_model_info",
            "Constant 1; the label tells which model version is loaded",
            ["version"],
            registry=self.registry,
        )
