import pytest

from spam_detector.evaluation import evaluate, metrics_at_threshold

Y = [1, 1, 0, 0, 0, 1]
P = [0.9, 0.6, 0.7, 0.2, 0.1, 0.4]


def test_metrics_at_threshold_counts():
    m = metrics_at_threshold(Y, P, 0.5)
    # predictions: 1, 1, 1, 0, 0, 0 -> TP=2, FP=1, FN=1, TN=2
    assert (m["true_positives"], m["false_positives"]) == (2, 1)
    assert (m["false_negatives"], m["true_negatives"]) == (1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_higher_threshold_trades_recall_for_precision():
    low = metrics_at_threshold(Y, P, 0.3)
    high = metrics_at_threshold(Y, P, 0.8)
    assert high["recall"] <= low["recall"]
    assert high["false_positives"] <= low["false_positives"]


def test_evaluate_returns_pr_auc_and_sweep():
    result = evaluate(Y, P, grid=(0.3, 0.5))
    assert 0.0 <= result["pr_auc"] <= 1.0
    assert len(result["threshold_sweep"]) == 2


def test_perfect_scores_give_pr_auc_one():
    assert evaluate([0, 1], [0.1, 0.9])["pr_auc"] == pytest.approx(1.0)
