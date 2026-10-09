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

The multi-stage image downloads the dataset (checksum-verified), trains the model in a build
stage, and ships only the virtualenv and the trained artifact in a slim runtime image. The
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

Known limitation: the Docker image still trains the word-n-gram baseline at build time
(`scripts/train.py`), because the registry lives in a local SQLite file. Serving the registry
champion from the image needs a shared tracking server, which is out of scope for a
zero-cost setup. Also, comparing several candidates on one test set slightly inflates the
winner's score; a production setup would select on a validation split and keep the test set
for the final check.
