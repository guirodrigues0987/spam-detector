# spam-detector

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

## API

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
