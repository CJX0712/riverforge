"""学习器 + 漂移探测器的数值不变量单测（硬金标准）。"""

from __future__ import annotations

import numpy as np
import pytest

from riverforge.core.config import Config
from riverforge.domain.streaming.detectors import ADWINDetector, NumpyDDM, make_detector
from riverforge.domain.streaming.numpy_impl import (
    NumpyOnlineLogistic,
    NumpyPA,
    NumpyPerceptron,
)


# ---------------- 数据工具 ----------------
def separable_stream(n=300, d=4, seed=0, margin=1.5):
    """线性可分：类1 沿 +w 方向偏移，类0 沿 -w 方向。"""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w = w / np.linalg.norm(w)
    for i in range(n):
        y = i % 2
        x = w * (margin if y == 1 else -margin) + rng.normal(0, 0.35, size=d)
        yield [float(v) for v in x], int(y)


def prequential_acc(learner, stream) -> float:
    ok = 0
    n = 0
    for x, y in stream:
        p = int(learner.predict_one(x))
        ok += int(p == y)
        n += 1
        learner.learn_one(x, y)
    return ok / max(n, 1)


# ---------------- 感知机 ----------------
def test_perceptron_update_moves_score_toward_correct_sign():
    p = NumpyPerceptron(lr=0.1, averaged=False)
    x = [1.0, 0.0]
    s0 = p.score_one(x)  # 初始 0
    p.learn_one(x, 1)
    s1 = p.score_one(x)
    assert s1 > s0, "正类样本更新后分数应变正"
    q = NumpyPerceptron(lr=0.1, averaged=False)
    q.learn_one(x, 0)
    assert q.score_one(x) < 0.0, "负类样本更新后分数应变负"


def test_perceptron_converges_on_separable():
    acc = prequential_acc(NumpyPerceptron(averaged=True), list(separable_stream(n=400)))
    assert acc > 0.90, f"感知机在线性可分流上应收敛, got {acc:.3f}"


def test_averaged_perceptron_runs_and_bounded():
    acc = prequential_acc(NumpyPerceptron(averaged=False), list(separable_stream(n=300)))
    assert 0.0 <= acc <= 1.0


# ---------------- PA ----------------
def test_pa_satisfies_unit_margin_after_update():
    """PA-I 不变量：更新后该样本间隔恰好达到 1（tau=loss/(||x||²+1)）。"""
    pa = NumpyPA()
    x = [1.0, 2.0]
    pa.learn_one(x, 1)
    margin = 1.0 * pa.score_one(x)  # y=1 -> s=+1
    assert abs(margin - 1.0) < 1e-6, f"PA-I 间隔应为 1, got {margin}"


def test_pa_no_update_when_margin_already_satisfied():
    pa = NumpyPA()
    x = [1.0, 0.0]
    pa.learn_one(x, 1)  # 第一次更新到 margin=1
    w_before = pa.w.copy()
    pa.learn_one(x, 1)  # 已满足 -> 不再更新
    assert np.allclose(w_before, pa.w)


def test_pa_converges_on_separable():
    acc = prequential_acc(NumpyPA(), list(separable_stream(n=400)))
    assert acc > 0.85, f"PA 应收敛, got {acc:.3f}"


# ---------------- 在线逻辑回归 ----------------
def test_online_logistic_probability_bounds():
    ol = NumpyOnlineLogistic()
    for x, y in separable_stream(n=60):
        p = ol.proba_one(x)
        assert 0.0 <= p <= 1.0
        ol.learn_one(x, y)


def test_online_logistic_converges_and_loss_drops():
    """可分数据上：后 1/3 的错误率应低于前 1/3（学习确实发生）。"""
    data = list(separable_stream(n=600))
    ol = NumpyOnlineLogistic(lr=0.1)
    errs = []
    for x, y in data:
        p = int(ol.predict_one(x))
        errs.append(int(p != y))
        ol.learn_one(x, y)
    first = np.mean(errs[:200])
    last = np.mean(errs[-200:])
    assert last < first, f"学习应降低错误率: first={first:.3f} last={last:.3f}"
    assert last < 0.15


# ---------------- DDM ----------------
def test_ddm_no_detection_when_no_errors():
    d = NumpyDDM(min_instances=30)
    assert not any(d.update(0) for _ in range(300))


def test_ddm_no_false_alarm_during_warmup():
    """回归测试：曾因预热期更新 p_min 导致每 min_instances 步误报一次。"""
    rng = np.random.default_rng(0)
    d = NumpyDDM(min_instances=40)
    fires = sum(1 for _ in range(400) if d.update(1 if rng.random() < 0.3 else 0))
    assert fires <= 2, f"恒定 30% 错误率不应频繁报警, got {fires}"


def test_ddm_detects_error_jump():
    d = NumpyDDM(min_instances=40)
    fired_at = None
    for i in range(400):
        err = 0 if i < 200 else 1  # 前 200 全对，之后全错
        if d.update(err):
            fired_at = i
            break
    assert fired_at is not None, "错误率骤升应被检出"
    assert fired_at >= 200


def test_ddm_reset_keeps_detection_count():
    d = NumpyDDM(min_instances=40)
    for i in range(400):
        d.update(0 if i < 200 else 1)
    assert d.n_detections >= 1


# ---------------- ADWIN（river 可选）----------------
@pytest.mark.skipif(not ADWINDetector.available(), reason="river 不可用")
def test_adwin_detects_error_jump():
    det = ADWINDetector(delta=0.002)
    fired = False
    for i in range(600):
        err = 0 if i < 200 else 1
        if det.update(err):
            fired = True
            break
    assert fired, "ADWIN 应检出错误率骤升"


@pytest.mark.skipif(not ADWINDetector.available(), reason="river 不可用")
def test_adwin_no_false_alarm_on_stable():
    det = ADWINDetector(delta=0.002)
    fired = any(det.update(0) for _ in range(600))
    assert not fired


# ---------------- 工厂 ----------------
def test_make_detector_ddm_always_available():
    d = make_detector("ddm", Config())
    assert isinstance(d, NumpyDDM)


def test_make_detector_auto_falls_back():
    d = make_detector("auto", Config())
    assert d.name in ("adwin", "ddm")


def test_make_detector_unknown_raises():
    with pytest.raises(ValueError):
        make_detector("nope", Config())
