"""Send test-set messages to a running API, optionally shifted to simulate data drift.

Examples:
    python scripts/simulate_traffic.py --n 300                # normal traffic
    python scripts/simulate_traffic.py --n 300 --drift        # shifted traffic

Uses only the standard library, so it runs anywhere the project's venv does.
"""

import argparse
import json
import os
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

from spam_detector.data import DATA_FILE, deduplicate, load_dataset, split_dataset

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
SEED = int(os.getenv("RANDOM_SEED", "42"))


def shift(text: str) -> str:
    """A deliberate change in the input distribution: longer, shouty, link- and digit-heavy."""
    text = text[:400].upper()  # stay under the API's 1,000-character limit after doubling
    return f"{text} {text} VISIT http://promo.example/{random.randint(10**8, 10**9)}"


def post(url: str, text: str) -> dict:
    request = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200, help="number of requests")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    parser.add_argument("--drift", action="store_true", help="shift the messages")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    _, test = split_dataset(deduplicate(load_dataset(DATA_DIR / DATA_FILE)), seed=SEED)
    texts = test["text"].sample(n=args.n, replace=args.n > len(test), random_state=args.seed)

    labels = Counter()
    for text in texts:
        labels[post(args.url, shift(text) if args.drift else text)["label"]] += 1
        time.sleep(args.delay)

    mode = "DRIFTED" if args.drift else "normal"
    print(f"sent {args.n} {mode} requests: {dict(labels)}")


if __name__ == "__main__":
    main()
