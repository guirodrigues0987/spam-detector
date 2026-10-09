import json
import logging

from spam_detector.logging_config import JsonFormatter, configure_logging


def make_record(**extra):
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_outputs_valid_json_with_extras():
    payload = json.loads(JsonFormatter().format(make_record(request_id="r1", status=200)))
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "r1"
    assert payload["status"] == 200
    assert "timestamp" in payload


def test_formatter_includes_exception_info():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_does_not_duplicate_handlers():
    configure_logging("INFO")
    configure_logging("INFO")
    json_handlers = [
        h for h in logging.getLogger().handlers if isinstance(h.formatter, JsonFormatter)
    ]
    assert len(json_handlers) == 1
