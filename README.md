# spam-detector

[![CI](https://github.com/guirodrigues0987/spam-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/guirodrigues0987/spam-detector/actions/workflows/ci.yml)

Production-grade SMS spam detection service. Work in progress: the model and evaluation
are done; API, Docker, CI/CD, deploy, MLOps and monitoring are next.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -e ".[dev]" && python scripts/run_all.py
```

On Linux/macOS use `source .venv/bin/activate` instead of `.venv/Scripts/activate`.

`run_all.py` downloads the dataset (checksum-verified), trains the model and evaluates it.
Artifacts land in `artifacts/`: `model.joblib`, `metadata.json`, `metrics.json`, `pr_curve.png`.

Run the tests and the linter with `pytest` and `ruff check .`.

## Run with Docker (one command)

```bash
docker compose up --build
```

The multi-stage image downloads the dataset (checksum-verified), trains the champion model
(character n-grams) in a build stage, and ships only the virtualenv and the trained artifact in a slim runtime image. The
container runs as a non-root user, with a read-only filesystem, all Linux capabilities dropped
and a `HEALTHCHECK` on `/health`. The API is then available on `http://127.0.0.1:8000`.
Tunable at runtime through `SPAM_THRESHOLD` and `LOG_LEVEL` (see `.env.example`).

### Prebuilt image

Every push to `main` that passes CI publishes the image to GitHub Container Registry, so
nothing needs to be built locally:

```bash
docker run --rm -p 8000:8000 ghcr.io/guirodrigues0987/spam-detector:latest
```

Images are tagged `latest` and with the commit SHA.

## API (without Docker)

After training, start the service (access logs are disabled because the app already emits
one structured JSON log line per request):

```bash
uvicorn spam_detector.api:app --port 8000 --no-access-log
```

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "WINNER! Claim your free prize now, text WIN to 80082"}'
# {"label":"spam","spam_probability":0.979,"threshold":0.5,"model_version":"..."}

curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"model_version":"..."}
```

- `POST /predict` validates input (non-blank text, max 1,000 chars; otherwise `422`).
- `GET /health` returns `503` if the model could not be loaded, so an orchestrator can
  see the instance is unhealthy.
- Logs are JSON on stdout. Message text is never logged (it may contain personal data);
  only its length and the predicted probability are.

## Model

- **Data:** [SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
  5,572 messages. 403 exact duplicates are removed *before* the split so copies cannot leak
  from train to test. Spam is ~12.6% of the data.
- **Model:** scikit-learn `Pipeline` of TF-IDF (unigrams + bigrams) and logistic regression with
  `class_weight="balanced"`. Stratified 80/20 split, fixed seed.
- **Reproducibility:** the dataset SHA-256 is pinned, and `metadata.json` records the seed,
  library versions, dataset hash and hyperparameters of every trained model.

## Results (held-out test set, 1,034 messages)

PR-AUC: **0.958**

| Threshold | Precision | Recall | F1    | False positives | False negatives |
|-----------|-----------|--------|-------|-----------------|-----------------|
| 0.3       | 0.714     | 0.954  | 0.817 | 50              | 6               |
| 0.5       | 0.902     | 0.916  | 0.909 | 13              | 11              |
| 0.7       | 0.965     | 0.832  | 0.893 | 4               | 22              |
| 0.9       | 1.000     | 0.527  | 0.690 | 0               | 62              |

## Why precision and recall are not equally important

The two errors have different costs:

- **False positive** (a legitimate message flagged as spam): the user may miss a bank code, a
  delivery notice or a message from a person. This is usually the more expensive error.
- **False negative** (spam reaches the inbox): annoying, occasionally dangerous (phishing),
  but the user can still ignore or report it.

Because of that, accuracy is not used (a model that always says "not spam" scores ~87%) and the
decision threshold is **configurable** (`SPAM_THRESHOLD`, default 0.5) instead of hard-coded.
Raising it trades recall for precision: at 0.7 the model produces 4 false positives instead of
13, at the price of 11 more missed spam messages. The right value depends on the product, and
the table above is what to look at when choosing it.

## Retraining with a quality gate (MLflow)

`scripts/retrain.py` trains a candidate, tracks it in MLflow and promotes it **only if it beats
the current champion**. It needs the optional `mlops` extra, which is not installed in the
production image: `pip install -e ".[mlops]"`.

```bash
python scripts/retrain.py                        # first run: becomes the champion
python scripts/retrain.py --analyzer char_wb     # challenger: character n-grams
mlflow ui --backend-store-uri sqlite:///mlflow.db    # browse runs and the model registry
```

- **Tracking:** every run logs parameters, metrics, the dataset SHA-256 and the git commit.
- **Registry:** every candidate is registered as a model version, with a `gate` tag
  (`promoted` / `rejected`) and the reason, so rejected attempts stay as an audit trail. The
  `champion` alias marks the version that is served.
- **Gate** (`src/spam_detector/promotion.py`): the candidate must improve PR-AUC by at least
  0.002 *and* lose no more than 0.02 precision at the serving threshold, because a false
  positive is the expensive error. The champion is re-scored on the same test split, so the
  comparison is fair.
- **Export:** the champion is written to `artifacts/`, which is what the API loads.

Example run history on this dataset:

| Run | PR-AUC | Precision | Gate |
|-----|--------|-----------|------|
| v1 baseline (word n-grams) | 0.9579 | 0.9023 | promoted (no champion yet) |
| v2 `C=0.01` | 0.9365 | 0.8667 | rejected (PR-AUC -0.0214) |
| v3 character n-grams | 0.9876 | 0.9612 | promoted (+0.0297) |
| v4 character n-grams, `C=10` | 0.9887 | 0.9690 | rejected (+0.0011 < 0.002) |

The Docker image does not read the registry (it lives in a local SQLite file, and sharing it
would need a tracking server, out of scope for a zero-cost setup). Instead it trains the
configuration the gate promoted, character n-grams, through `scripts/train.py --analyzer
char_wb`, so the image serves the same model as the champion above (PR-AUC 0.9876). The choice
is a build argument: `docker build --build-arg ANALYZER=word .` builds the baseline. If a
future run promotes a different configuration, the Dockerfile default has to be updated by hand.
Also, comparing several candidates on one test set slightly inflates the
winner's score; a production setup would select on a validation split and keep the test set
for the final check.

## Monitoring

```bash
docker compose --profile monitoring up --build
```

| Service | URL | Notes |
|---------|-----|-------|
| API | http://127.0.0.1:8000 | public port; metrics are **not** served here |
| Prometheus | http://127.0.0.1:9090 | scrapes the API every 15 s |
| Grafana | http://127.0.0.1:3000 | dashboard **Spam Detector** is provisioned from files |

All ports are bound to `127.0.0.1`. Grafana's login is `admin` / `admin` unless you set
`GRAFANA_ADMIN_PASSWORD` in `.env` (see `.env.example`). Generate some traffic to see the
dashboard move: `python scripts/simulate_traffic.py --n 300`.

**Where metrics are served:** on their own port, not on the API's. Setting `METRICS_PORT`
starts a Prometheus endpoint (`/metrics`) on it, bound to `METRICS_HOST` (default `127.0.0.1`);
leaving it unset starts nothing. `docker-compose.yml` sets port 9100 and only exposes it to the
compose network, so Prometheus can scrape `api:9100` but neither the host nor the internet can.
A deploy that publishes a single port, like the Hugging Face Space, never exposes telemetry,
and `/metrics` on the API port returns 404.

**Metrics** (`src/spam_detector/metrics.py`): request count by route and status, latency
histogram (for p50/p95/p99), predictions by class (the predicted spam rate), a histogram of
predicted probabilities, and the loaded model version. Routes are labelled by their template,
and unknown URLs share a single `unmatched` label, so a scanner cannot blow up the number of
time series.

**Dashboard panels:** requests/s, latency p95, predicted spam rate, 5xx rate, requests by
status, latency percentiles, predictions by class, median predicted probability.

### Drift report (Evidently)

Each prediction is logged with anonymous numeric features of the message (length, word count,
digit ratio, uppercase ratio, has-URL) and the predicted probability. **The text is never
logged**, and these summaries cannot be turned back into it. `scripts/drift_report.py`
compares them against the training data:

```bash
pip install -e ".[monitoring]"
python scripts/simulate_traffic.py --n 300 --drift          # shifted traffic, for the demo
docker compose logs api --no-log-prefix > logs/api.log
python scripts/drift_report.py --logs logs/api.log          # writes reports/drift.html
```

On this dataset, normal traffic shows no drift (0 of 6 columns; every distance is below the
0.1 threshold), while the shifted traffic flags all 6. Use `--fail-on-drift` to exit non-zero,
for example from a scheduled job. The model in `ARTIFACTS_DIR` must be the one that served the
logs, since the predicted-probability column is compared against its scores on the reference
data.
