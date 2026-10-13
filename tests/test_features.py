import logging

import pytest

from spam_detector.features import FEATURE_NAMES, text_features
from tests.conftest import SPAM_TEXT


def test_features_of_a_plain_message():
    f = text_features("Hello World 123")
    assert f["text_length"] == 15
    assert f["word_count"] == 3
    assert f["digit_ratio"] == pytest.approx(3 / 15, abs=1e-4)
    assert f["uppercase_ratio"] == pytest.approx(2 / 10, abs=1e-4)  # H, W of 10 letters
    assert f["has_url"] == 0


@pytest.mark.parametrize("text", ["see www.example.com now", "go to HTTP://x.co", "https://a.b"])
def test_url_detection(text):
    assert text_features(text)["has_url"] == 1


@pytest.mark.parametrize("text", ["", " ", "😀😀", "12345", "ÀÇÃO"])
def test_features_never_divide_by_zero(text):
    assert set(text_features(text)) == set(FEATURE_NAMES)


def test_features_do_not_contain_the_message(client, caplog):
    secret = "my code is 998877, claim your free prize"
    with caplog.at_level(logging.INFO, logger="spam_detector.api"):
        client.post("/predict", json={"text": secret})

    record = next(r for r in caplog.records if r.getMessage() == "prediction")
    for name in FEATURE_NAMES:
        assert hasattr(record, name)
    assert secret not in str(record.__dict__)
    assert "998877" not in str(record.__dict__)
    assert SPAM_TEXT not in str(record.__dict__)
