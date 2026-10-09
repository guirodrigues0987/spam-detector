"""Runtime configuration read from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path
    spam_threshold: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
            spam_threshold=float(os.getenv("SPAM_THRESHOLD", "0.5")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
