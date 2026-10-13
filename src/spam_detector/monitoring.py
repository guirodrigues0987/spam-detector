"""Data-drift monitoring: build a dataset from the API's prediction logs and compare it to a
reference sample with Evidently.

Only `build_drift_report` needs the optional `monitoring` extra; everything else is plain
pandas, so it is cheap to test.
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from spam_detector.features import FEATURE_NAMES, text_features

COLUMNS = [*FEATURE_NAMES, "spam_probability"]

_DRIFT_METRIC = re.compile(
    r"ValueDrift\(column=(?P<column>.+?),method=(?P<method>.+?),threshold=(?P<threshold>[\d.eE+-]+)\)"
)


def read_log_lines(path: Path) -> list[str]:
    """Read a log file, coping with UTF-16 (what PowerShell's `>` writes on Windows)."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16").splitlines()
    return raw.decode("utf-8-sig", errors="replace").splitlines()


def parse_prediction_logs(lines: Iterable[str]) -> pd.DataFrame:
    """Extract the anonymous per-prediction records from the API's JSON log lines.

    Non-JSON lines (uvicorn banners, tracebacks) and non-prediction records are skipped.
    """
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("message") != "prediction" or not all(c in record for c in COLUMNS):
            continue
        rows.append({c: record[c] for c in COLUMNS})
    return pd.DataFrame(rows, columns=COLUMNS)


def reference_frame(texts: Iterable[str], probabilities: Iterable[float]) -> pd.DataFrame:
    """Same columns as the logged data, computed from the reference texts and model scores."""
    frame = pd.DataFrame([text_features(t) for t in texts], columns=FEATURE_NAMES)
    frame["spam_probability"] = list(probabilities)
    return frame


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Run Evidently's data-drift preset; returns the snapshot (`.dict()`, `.save_html()`)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=COLUMNS)
    report = Report([DataDriftPreset()])
    return report.run(
        Dataset.from_pandas(current[COLUMNS], data_definition=definition),
        Dataset.from_pandas(reference[COLUMNS], data_definition=definition),
    )


def summarize_drift(result: dict) -> dict:
    """Turn Evidently's `snapshot.dict()` into {drifted_share, columns: {name: {...}}}."""
    columns, share = {}, 0.0
    for metric in result["metrics"]:
        name, value = metric["metric_name"], metric["value"]
        if name.startswith("DriftedColumnsCount"):
            share = value["share"]
            continue
        match = _DRIFT_METRIC.match(name)
        if not match:
            continue
        threshold = float(match["threshold"])
        # Statistical tests report a p-value (drift when small); distances drift when large.
        drifted = value < threshold if "p_value" in match["method"] else value > threshold
        columns[match["column"]] = {
            "drifted": bool(drifted),
            "method": match["method"],
            "value": float(value),
        }
    return {"drifted_share": share, "columns": columns}
