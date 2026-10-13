"""FastAPI application: /predict and /health (metrics live on a separate port)."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import start_http_server

from spam_detector.config import Settings
from spam_detector.features import text_features
from spam_detector.logging_config import configure_logging
from spam_detector.metrics import Metrics
from spam_detector.model import load_artifact, load_metadata
from spam_detector.schemas import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger("spam_detector.api")

# Probes run every few seconds; logging them at INFO would drown real traffic.
QUIET_PATHS = {"/health"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.model_version = None
        app.state.metrics = metrics
        app.state.metrics_server = None
        if settings.metrics_port is not None:
            app.state.metrics_server, _ = start_http_server(
                settings.metrics_port, addr=settings.metrics_host, registry=metrics.registry
            )
            logger.info("metrics server started", extra={"metrics_port": settings.metrics_port})
        try:
            app.state.model = load_artifact(settings.artifacts_dir)
            app.state.model_version = load_metadata(settings.artifacts_dir)["trained_at"]
            metrics.model_info.labels(version=app.state.model_version).set(1)
            logger.info("model loaded", extra={"model_version": app.state.model_version})
        except Exception:
            # Start anyway: /health reports 503 so the orchestrator sees an unhealthy instance.
            logger.exception(
                "failed to load model", extra={"artifacts_dir": str(settings.artifacts_dir)}
            )
        yield
        if app.state.metrics_server is not None:
            app.state.metrics_server.shutdown()
            app.state.metrics_server.server_close()

    app = FastAPI(title="spam-detector", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe_requests(request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["x-request-id"] = request_id

        # Label by route template, never by raw URL, to keep metric cardinality bounded.
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        metrics.requests.labels(request.method, path, response.status_code).inc()
        metrics.latency.labels(request.method, path).observe(elapsed)

        logger.log(
            logging.DEBUG if path in QUIET_PATHS else logging.INFO,
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed * 1000, 2),
            },
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request, response: Response) -> HealthResponse:
        loaded = request.app.state.model is not None
        if not loaded:
            response.status_code = 503
        return HealthResponse(
            status="ok" if loaded else "unavailable",
            model_loaded=loaded,
            model_version=request.app.state.model_version,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        proba = float(model.predict_proba([body.text])[0, 1])
        is_spam = proba >= settings.spam_threshold
        label = "spam" if is_spam else "ham"

        metrics.predictions.labels(label).inc()
        metrics.probability.observe(proba)
        # The message text is deliberately not logged: SMS may contain personal data. Only
        # anonymous numeric summaries are, and those feed the drift report.
        logger.info(
            "prediction",
            extra={
                **text_features(body.text),
                "spam_probability": round(proba, 4),
                "is_spam": is_spam,
            },
        )
        return PredictResponse(
            label=label,
            spam_probability=proba,
            threshold=settings.spam_threshold,
            model_version=request.app.state.model_version,
        )

    return app


app = create_app()
