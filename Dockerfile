# syntax=docker/dockerfile:1

# ---- Stage 1: build dependencies and train the model --------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Install the package (runtime dependencies only). Copying the metadata first keeps this
# layer cached until pyproject.toml or the source changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Download the dataset (checksum-verified) and train. The model is baked into the image.
COPY scripts ./scripts
RUN python scripts/download_data.py && python scripts/train.py

# ---- Stage 2: minimal runtime image -------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ARTIFACTS_DIR=/app/artifacts

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/artifacts ./artifacts

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "spam_detector.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
