"""流式评测指标（prequential：先测后学的交错评测口径）。

- accuracy / kappa：全流累计。
- auc：基于决策分数的 rank-AUC（处理并列）。
- drift_recall / drift_false_alarm：与真实漂移点比对（窗口容差）。
- cross_check_with_river：与顶级开源 river 的 Accuracy/CohenKappa 对拍（单测交叉验证）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import NumericalError


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均），mergesort 保证稳定。"""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auc_score(scores, labels) -> float:
    """rank-AUC（并列修正）；单类退化为 0.5。"""
    if len(scores) == 0:
        return 0.5
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(s)
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def cohen_kappa(y_true, y_pred) -> float:
    """Cohen's kappa（流式累计口径）。"""
    n = len(y_true)
    if n == 0:
        return 0.0
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    p_o = float((yt == yp).sum()) / n
    classes = np.unique(np.concatenate([yt, yp]))
    p_e = 0.0
    for c in classes:
        p_e += (float((yp == c).sum()) / n) * (float((yt == c).sum()) / n)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


class PrequentialEvaluator:
    """交错评测累加器：accuracy / kappa / auc / n。"""

    def __init__(self):
        self.y_true: list = []
        self.y_pred: list = []
        self.scores: list = []

    def update(self, y_true, y_pred, score=None) -> PrequentialEvaluator:
        self.y_true.append(int(y_true))
        self.y_pred.append(int(y_pred))
        self.scores.append(float(score) if score is not None else float(y_pred == 1))
        return self

    def get(self) -> dict:
        n = len(self.y_true)
        if n == 0:
            return {"accuracy": 0.0, "kappa": 0.0, "auc": 0.5, "n": 0}
        yt = np.asarray(self.y_true, dtype=int)
        yp = np.asarray(self.y_pred, dtype=int)
        acc = float((yt == yp).sum()) / n
        if acc < -1e-9 or acc > 1 + 1e-9:  # 不变量守护
            raise NumericalError(f"prequential accuracy 越界: {acc}")
        return {
            "accuracy": acc,
            "kappa": cohen_kappa(yt, yp),
            "auc": auc_score(self.scores, yt),
            "n": n,
        }


def drift_scores(detected_at, true_positions, n_total: int, window: int = 100) -> tuple:
    """(recall, false_alarm_rate)。

    recall：真实漂移点中被（窗口内）检出的比例。
    false_alarm：落在任何真实漂移窗口外的误报占全部报警的比例（无真实漂移时按步数归一）。
    """
    det = list(detected_at or [])
    pos = list(true_positions or [])
    if not pos:
        if not det:
            return 0.0, 0.0
        return 0.0, len(det) / max(n_total, 1)

    hits = 0
    for p in pos:
        if any(abs(d - p) <= window for d in det):
            hits += 1
    recall = hits / len(pos)

    false = sum(1 for d in det if not any(abs(d - p) <= window for p in pos))
    fa = false / max(len(det), 1)
    return recall, fa


def cross_check_with_river(y_true, y_pred):
    """与 river 的 Accuracy/CohenKappa 对拍（river 不可用时返回 None）。"""
    try:
        from river.metrics import Accuracy, CohenKappa  # type: ignore
    except Exception:
        return None
    acc, kappa = Accuracy(), CohenKappa()
    for t, p in zip(y_true, y_pred):
        acc.update(int(t), int(p))
        kappa.update(int(t), int(p))
    return {"accuracy": acc.get(), "kappa": kappa.get()}
