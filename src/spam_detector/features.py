"""Anonymous numeric features of a message, used for drift monitoring.

The text itself is never stored; these summaries cannot be turned back into the message.
"""

import re

_URL = re.compile(r"(https?://|www\.)", re.IGNORECASE)

FEATURE_NAMES = ["text_length", "word_count", "digit_ratio", "uppercase_ratio", "has_url"]


def text_features(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    return {
        "text_length": len(text),
        "word_count": len(text.split()),
        "digit_ratio": round(sum(c.isdigit() for c in text) / max(len(text), 1), 4),
        "uppercase_ratio": round(sum(c.isupper() for c in letters) / max(len(letters), 1), 4),
        "has_url": int(bool(_URL.search(text))),
    }
