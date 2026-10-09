"""Download the SMS Spam Collection (UCI) and verify its checksum."""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

from spam_detector.data import DATA_FILE, deduplicate, load_dataset

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
DATA_DIR = Path("data")


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / DATA_FILE
    if not target.exists():
        archive = DATA_DIR / "sms.zip"
        urllib.request.urlretrieve(URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != SHA256:
            archive.unlink()
            print(f"Checksum mismatch: expected {SHA256}, got {digest}", file=sys.stderr)
            return 1
        with zipfile.ZipFile(archive) as zf:
            zf.extract(DATA_FILE, DATA_DIR)
        archive.unlink()

    df = load_dataset(target)
    unique = deduplicate(df)
    print(f"rows: {len(df)}")
    print(f"duplicates removed: {len(df) - len(unique)}")
    print(f"spam ratio (deduplicated): {unique['label'].mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
