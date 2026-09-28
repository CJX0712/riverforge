"""评测指标单测：AUC / kappa / prequential / 漂移召回 / 与 river 交叉验证。"""

from __future__ import annotations

import pytest

from riverforge.eval.metrics import (
    PrequentialEvaluator,
    auc_score,
    cohen_kappa,
    cross_check_with_river,
    drift_scores,
)


def test_auc_perfect_separation():
    assert auc_score([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_auc_inverted():
    assert auc_score([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_auc_random_half():
    scores = [0.1, 0.2, 0.3, 0.4]
    labels = [0, 1, 1, 0]
    assert abs(auc_score(scores, labels) - 0.5) < 1e-9


def test_auc_all_ties_is_half():
    assert abs(auc_score([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) - 0.5) < 1e-9


def test_auc_single_class_returns_half():
    assert auc_score([0.1, 0.2, 0.3], [1, 1, 1]) == 0.5


def test_auc_ties_handled_partial():
    """部分并列：正类分 {1.0, 2.0}，负类分 {1.0, 3.0}。

    4 个 (pos, neg) 对：(1.0,1.0)=0.5, (1.0,3.0)=0, (2.0,1.0)=1, (2.0,3.0)=0
    -> (0.5+0+1+0)/4 = 0.375
    """
    a = auc_score([1.0, 1.0, 2.0, 3.0], [1, 0, 1, 0])
    assert abs(a - 0.375) < 1e-9


def test_kappa_perfect_is_one():
    y = [0, 1, 1, 0, 1]
    assert abs(cohen_kappa(y, y) - 1.0) < 1e-9


def test_kappa_bounded():
    k = cohen_kappa([0, 0, 1, 1], [1, 1, 0, 0])
    assert -1.0 <= k <= 1.0


def test_evaluator_accuracy_exact():
    ev = PrequentialEvaluator()
    pairs = [(1, 1), (0, 0), (1, 0), (0, 0)]  # 3/4 正确
    for t, p in pairs:
        ev.update(t, p, score=float(p))
    m = ev.get()
    assert abs(m["accuracy"] - 0.75) < 1e-9
    assert m["n"] == 4


def test_evaluator_empty_safe():
    assert PrequentialEvaluator().get()["n"] == 0


def test_drift_scores_hit():
    r, fa = drift_scores([510], [500], n_total=1000, window=150)
    assert r == 1.0 and fa == 0.0


def test_drift_scores_miss_and_false_alarm():
    # 检出点 900 远离真实漂移 500 -> 既漏报又误报
    r, fa = drift_scores([900], [500], n_total=1000, window=150)
    assert r == 0.0 and fa == 1.0


def test_drift_scores_no_detections_no_positions():
    assert drift_scores([], [], n_total=1000) == (0.0, 0.0)


def test_drift_scores_no_true_drift_but_detections():
    r, fa = drift_scores([10, 20], [], n_total=1000)
    assert r == 0.0 and fa > 0.0


def test_drift_scores_multiple_positions_partial():
    r, fa = drift_scores([260, 760], [250, 500, 750], n_total=1000, window=150)
    assert abs(r - 2 / 3) < 1e-9


@pytest.mark.skipif(cross_check_with_river([0], [0]) is None, reason="river 不可用")
def test_cross_check_accuracy_matches_river():
    """不变量：自研 numpy accuracy/kappa 必须与 river 的指标一致。"""
    rng_t = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    rng_p = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
    ev = PrequentialEvaluator()
    for t, p in zip(rng_t, rng_p):
        ev.update(t, p, score=float(p))
    mine = ev.get()
    ref = cross_check_with_river(rng_t, rng_p)
    assert ref is not None
    assert abs(mine["accuracy"] - ref["accuracy"]) < 1e-9
    assert abs(mine["kappa"] - ref["kappa"]) < 1e-9
