"""FastAPI application: /predict and /health."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from spam_detector.config import Settings
from spam_detector.logging_config import configure_logging
from spam_detector.model import load_artifact, load_metadata
from spam_detector.schemas import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger("spam_detector.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.model_version = None
        try:
            app.state.model = load_artifact(settings.artifacts_dir)
            app.state.model_version = load_metadata(settings.artifacts_dir)["trained_at"]
            logger.info("model loaded", extra={"model_version": app.state.model_version})
        except Exception:
            # Start anyway: /health reports 503 so the orchestrator sees an unhealthy instance.
            logger.exception(
                "failed to load model", extra={"artifacts_dir": str(settings.artifacts_dir)}
            )
        yield

    app = FastAPI(title="spam-detector", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
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
        # The message text is deliberately not logged: SMS may contain personal data.
        proba = float(model.predict_proba([body.text])[0, 1])
        is_spam = proba >= settings.spam_threshold
        logger.info(
            "prediction",
            extra={
                "text_length": len(body.text),
                "spam_probability": round(proba, 4),
                "is_spam": is_spam,
            },
        )
        return PredictResponse(
            label="spam" if is_spam else "ham",
            spam_probability=proba,
            threshold=settings.spam_threshold,
            model_version=request.app.state.model_version,
        )

    return app


app = create_app()
