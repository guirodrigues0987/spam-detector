import json

import numpy as np
import pandas as pd
import pytest

from spam_detector.features import FEATURE_NAMES
from spam_detector.monitoring import (
    COLUMNS,
    parse_prediction_logs,
    read_log_lines,
    reference_frame,
    summarize_drift,
)


def prediction_line(**overrides) -> str:
    record = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "level": "INFO",
        "message": "prediction",
        "text_length": 40,
        "word_count": 8,
        "digit_ratio": 0.05,
        "uppercase_ratio": 0.1,
        "has_url": 0,
        "spam_probability": 0.2,
        "is_spam": False,
        **overrides,
    }
    return json.dumps(record)


def test_parse_keeps_only_prediction_records():
    lines = [
        "INFO:     Started server process [1]",  # uvicorn banner
        json.dumps({"message": "request", "path": "/predict"}),
        prediction_line(text_length=10),
        "{broken json",
        prediction_line(text_length=20),
        "",
    ]
    frame = parse_prediction_logs(lines)
    assert list(frame.columns) == COLUMNS
    assert frame["text_length"].tolist() == [10, 20]


def test_parse_skips_records_missing_a_feature():
    record = json.loads(prediction_line())
    del record["word_count"]
    assert parse_prediction_logs([json.dumps(record)]).empty


def test_parse_of_empty_input_still_has_the_columns():
    assert list(parse_prediction_logs([]).columns) == COLUMNS


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_read_log_lines_handles_windows_powershell_utf16(tmp_path, encoding):
    path = tmp_path / "api.log"
    path.write_text(prediction_line() + "\n" + prediction_line() + "\n", encoding=encoding)
    assert len(parse_prediction_logs(read_log_lines(path))) == 2


def test_reference_frame_has_the_same_columns_as_the_logs():
    frame = reference_frame(["Hello World", "WIN http://x.co now"], [0.1, 0.9])
    assert list(frame.columns) == [*FEATURE_NAMES, "spam_probability"] == COLUMNS
    assert frame["has_url"].tolist() == [0, 1]
    assert frame["spam_probability"].tolist() == [0.1, 0.9]


def drift_metric(column: str, method: str, threshold: float, value: float) -> dict:
    name = f"ValueDrift(column={column},method={method},threshold={threshold})"
    return {"metric_name": name, "value": value}


def test_summarize_handles_p_value_and_distance_methods():
    wasserstein = "Wasserstein distance (normed)"
    result = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "value": {"count": 2.0, "share": 0.5},
            },
            drift_metric("a", "K-S p_value", 0.05, 0.001),
            drift_metric("b", "K-S p_value", 0.05, 0.4),
            drift_metric("c", wasserstein, 0.1, 0.5),
            drift_metric("d", wasserstein, 0.1, 0.02),
        ]
    }
    summary = summarize_drift(result)
    drifted = {name: info["drifted"] for name, info in summary["columns"].items()}
    assert drifted == {"a": True, "b": False, "c": True, "d": False}
    assert summary["drifted_share"] == 0.5


def test_evidently_flags_a_shifted_column_and_not_a_stable_one():
    pytest.importorskip("evidently", reason="needs the optional `monitoring` extra")
    from spam_detector.monitoring import build_drift_report

    rng = np.random.default_rng(0)

    def frame(shift: float) -> pd.DataFrame:
        data = {name: rng.normal(10, 1, 300) for name in COLUMNS}
        data["text_length"] = rng.normal(10 + shift, 1, 300)
        return pd.DataFrame(data)

    summary = summarize_drift(build_drift_report(frame(0), frame(5)).dict())
    assert summary["columns"]["text_length"]["drifted"]
    assert not summary["columns"]["word_count"]["drifted"]
