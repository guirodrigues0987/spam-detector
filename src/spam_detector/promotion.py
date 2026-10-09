"""Quality gate: decide whether a freshly trained model may replace the champion."""

from dataclasses import dataclass

MIN_IMPROVEMENT = 0.002  # required PR-AUC gain over the champion
MAX_PRECISION_DROP = 0.02  # false positives are the expensive error, so guard precision


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str


def decide(
    candidate: dict,
    champion: dict | None,
    min_improvement: float = MIN_IMPROVEMENT,
    max_precision_drop: float = MAX_PRECISION_DROP,
) -> Decision:
    """Compare two metric dicts (keys: `pr_auc`, `precision`) measured on the same test set.

    The candidate must beat the champion's PR-AUC by `min_improvement` and must not lose more
    than `max_precision_drop` of precision at the serving threshold.
    """
    if champion is None:
        return Decision(True, "no champion yet: the first model becomes the baseline")

    gain = candidate["pr_auc"] - champion["pr_auc"]
    if gain < min_improvement:
        return Decision(
            False, f"PR-AUC gain {gain:+.4f} is below the required {min_improvement:+.4f}"
        )

    drop = champion["precision"] - candidate["precision"]
    if drop > max_precision_drop:
        return Decision(
            False,
            f"precision dropped by {drop:.4f}, more than the allowed {max_precision_drop:.4f}",
        )

    return Decision(True, f"PR-AUC improved by {gain:+.4f} and precision is within tolerance")
