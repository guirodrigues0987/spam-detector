"""Runtime configuration read from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path
    spam_threshold: float
    log_level: str
    # Prometheus metrics are served on their own port so they never share the public one.
    # None (the default) means no metrics server at all.
    metrics_port: int | None = None
    metrics_host: str = "127.0.0.1"

    @classmethod
    def from_env(cls) -> "Settings":
        metrics_port = os.getenv("METRICS_PORT", "").strip()
        return cls(
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
            spam_threshold=float(os.getenv("SPAM_THRESHOLD", "0.5")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            metrics_port=int(metrics_port) if metrics_port else None,
            metrics_host=os.getenv("METRICS_HOST", "127.0.0.1"),
        )
