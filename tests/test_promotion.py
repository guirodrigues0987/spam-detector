import pytest

from spam_detector.promotion import decide

CHAMPION = {"pr_auc": 0.95, "precision": 0.90}


def test_first_model_is_promoted_when_there_is_no_champion():
    decision = decide({"pr_auc": 0.5, "precision": 0.5}, None)
    assert decision.promote
    assert "baseline" in decision.reason


def test_clear_improvement_is_promoted():
    decision = decide({"pr_auc": 0.97, "precision": 0.91}, CHAMPION)
    assert decision.promote


def test_equal_model_is_rejected():
    decision = decide(dict(CHAMPION), CHAMPION)
    assert not decision.promote
    assert "PR-AUC" in decision.reason


def test_worse_model_is_rejected():
    assert not decide({"pr_auc": 0.90, "precision": 0.95}, CHAMPION).promote


def test_gain_below_minimum_is_rejected():
    candidate = {"pr_auc": 0.9505, "precision": 0.90}
    assert not decide(candidate, CHAMPION, min_improvement=0.002).promote
    assert decide(candidate, CHAMPION, min_improvement=0.0001).promote


def test_better_pr_auc_but_much_lower_precision_is_rejected():
    decision = decide({"pr_auc": 0.99, "precision": 0.80}, CHAMPION)
    assert not decision.promote
    assert "precision" in decision.reason


def test_small_precision_drop_is_tolerated():
    assert decide({"pr_auc": 0.97, "precision": 0.89}, CHAMPION).promote


@pytest.mark.parametrize(("precision", "expected"), [(0.881, True), (0.879, False)])
def test_precision_tolerance_boundary(precision, expected):
    candidate = {"pr_auc": 0.97, "precision": precision}
    assert decide(candidate, CHAMPION, max_precision_drop=0.02).promote is expected
