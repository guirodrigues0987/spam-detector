"""Evaluation metrics for the spam classifier (positive class = spam)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    """Precision/recall/F1 plus raw error counts at a given decision threshold."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positives": int(fp),  # legitimate messages sent to spam
        "false_negatives": int(fn),  # spam that reached the inbox
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def evaluate(y_true, proba, threshold: float = 0.5, grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Headline metrics at `threshold`, threshold-free PR-AUC, and a threshold sweep."""
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "at_threshold": metrics_at_threshold(y_true, proba, threshold),
        "threshold_sweep": [metrics_at_threshold(y_true, proba, t) for t in grid],
    }
