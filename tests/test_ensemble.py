"""旗舰 DriftForge 与消融基线：自适应性 + 漂移检出 + 不变量。"""

from __future__ import annotations

from riverforge.core.config import Config
from riverforge.data.stream import make_dataset
from riverforge.domain.streaming.ensemble import DriftForge, DriftResetWrapper
from riverforge.domain.streaming.numpy_impl import NumpyPerceptron
from riverforge.eval.metrics import drift_scores


def _run(learner, ds):
    """prequential 跑一个流，返回 (accuracy, learner)。"""
    ok = n = 0
    for s in ds.stream():
        ok += int(int(learner.predict_one(s.x)) == int(s.y))
        n += 1
        learner.learn_one(s.x, s.y)
    return ok / max(n, 1), learner


def _cfg(n=1000):
    c = Config()
    c.n_samples = n
    return c


def test_driftforge_weights_are_a_distribution():
    df = DriftForge(cfg=_cfg(200))
    for i in range(30):
        df.learn_one([0.1 * i] * 6, i % 2)
    assert len(df.weights) == len(df.learners)
    assert all(w >= 0 for w in df.weights)
    assert abs(sum(df.weights) - 1.0) < 1e-9


def test_driftforge_accuracy_in_range():
    cfg = _cfg(400)
    ds = make_dataset("s", "sudden", cfg)
    acc, _ = _run(DriftForge(cfg=cfg), ds)
    assert 0.0 <= acc <= 1.0


def test_driftforge_beats_static_perceptron_on_sudden_drift():
    """核心不变量：漂移场景下旗舰显著优于不重置的静态感知机。"""
    cfg = _cfg(1000)
    ds_static = make_dataset("s", "sudden", cfg)
    ds_flag = make_dataset("s", "sudden", cfg)

    static_acc, _ = _run(NumpyPerceptron(averaged=True), ds_static)
    flag_acc, _ = _run(DriftForge(cfg=cfg), ds_flag)
    assert flag_acc > static_acc + 0.05, (
        f"旗舰应显著优于静态感知机: flag={flag_acc:.3f} static={static_acc:.3f}"
    )


def test_driftforge_detects_sudden_drift():
    cfg = _cfg(1000)
    ds = make_dataset("s", "sudden", cfg)
    _, df = _run(DriftForge(cfg=cfg), ds)
    recall, _ = drift_scores(
        df.detected_at, ds.drift_positions, cfg.n_samples, window=cfg.drift_window
    )
    assert recall >= 1.0, f"应检出突发漂移, detected_at={df.detected_at}"


def test_driftforge_low_false_alarm_on_stationary():
    cfg = _cfg(1000)
    ds = make_dataset("s", "none", cfg)
    _, df = _run(DriftForge(cfg=cfg), ds)
    assert len(df.detected_at) <= 2, f"平稳流不应频繁误报: {df.detected_at}"


def test_driftforge_no_false_alarm_on_noisy_stationary():
    cfg = _cfg(1000)
    ds = make_dataset("noisy", "noisy", cfg)
    _, df = _run(DriftForge(cfg=cfg), ds)
    assert len(df.detected_at) <= 3


def test_reset_wrapper_detects_and_resets_on_drift():
    cfg = _cfg(1000)
    ds = make_dataset("s", "sudden", cfg)
    w = DriftResetWrapper(learner=NumpyPerceptron(averaged=True), cfg=cfg)
    acc, w = _run(w, ds)
    assert len(w.detected_at) >= 1, "DDM 应在突发漂移上报警"
    assert 0.0 <= acc <= 1.0


def test_reset_wrapper_improves_over_static():
    cfg = _cfg(1000)
    static_acc, _ = _run(NumpyPerceptron(averaged=True), make_dataset("s", "sudden", cfg))
    reset_acc, _ = _run(
        DriftResetWrapper(learner=NumpyPerceptron(averaged=True), cfg=cfg),
        make_dataset("s", "sudden", cfg),
    )
    assert reset_acc > static_acc + 0.05, (
        f"漂移重置应显著优于静态: reset={reset_acc:.3f} static={static_acc:.3f}"
    )


def test_driftforge_reset_clears_state():
    df = DriftForge(cfg=_cfg(200))
    for i in range(50):
        df.learn_one([0.2] * 6, i % 2)
    df = df.reset()
    assert df.t == 0
    assert df.detected_at == []
    assert len(df.ema_err) == len(df.learners)


def test_driftforge_ema_bounded():
    df = DriftForge(cfg=_cfg(200))
    for i in range(100):
        df.learn_one([0.1 * (i % 7)] * 6, i % 2)
    assert all(0.0 <= e <= 1.0 for e in df.ema_err)
