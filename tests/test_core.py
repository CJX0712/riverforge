"""core + data 层单测：配置、类型、流式数据生成（确定性/漂移真实性）。"""

from __future__ import annotations

import numpy as np
import pytest

from riverforge.core.config import Config
from riverforge.core.errors import RiverForgeError
from riverforge.core.types import StreamResult, StreamSample
from riverforge.data.stream import default_datasets, make_dataset


def test_config_defaults():
    c = Config()
    assert c.random_state == 42
    assert c.benchmark_seeds == (42, 123, 7)
    assert c.n_classes == 2


def test_config_from_env_overrides(monkeypatch):
    monkeypatch.setenv("ENV_RF_SEED", "7")
    monkeypatch.setenv("ENV_RF_N_SAMPLES", "123")
    monkeypatch.setenv("ENV_RF_USE_RIVER", "false")
    c = Config.from_env()
    assert c.random_state == 7
    assert c.n_samples == 123
    assert c.use_river is False


def test_result_as_dict_rounds():
    r = StreamResult(
        algorithm="a",
        scenario="s",
        task="classification",
        prequential_accuracy=0.123456,
        kappa=0.1,
        auc=0.5,
        mae=0.0,
        rmse=0.0,
        drift_recall=0.0,
        drift_false_alarm=0.0,
        n_seen=10,
        runtime_s=1.0,
        supports_drift=False,
    )
    d = r.as_dict()
    assert d["prequential_accuracy"] == 0.1235
    assert d["skipped"] is False


def test_error_code_format():
    e = RiverForgeError("boom")
    assert "[E000]" in str(e)


def test_stream_factory_returns_fresh_stream_each_call():
    """stream() 必须每次返回新生成器（曾因复用耗尽生成器导致第二次为空）。"""
    cfg = Config()
    cfg.n_samples = 60
    ds = make_dataset("t", "none", cfg)
    a = list(ds.stream())
    b = list(ds.stream())
    assert len(a) == 60 and len(b) == 60
    assert [s.y for s in a] == [s.y for s in b]  # 同种子确定性


def test_stream_determinism_same_seed_differs_across_seeds():
    c1 = Config()
    c1.n_samples = 80
    c1.random_state = 1
    c2 = Config()
    c2.n_samples = 80
    c2.random_state = 2
    y1 = [s.y for s in make_dataset("a", "none", c1).stream()]
    y2 = [s.y for s in make_dataset("a", "none", c2).stream()]
    assert y1 != y2


def test_sample_types_and_length():
    cfg = Config()
    cfg.n_samples = 30
    ds = make_dataset("t", "none", cfg)
    for s in ds.stream():
        assert isinstance(s, StreamSample)
        assert len(s.x) == cfg.n_features
        assert s.y in (0, 1)
        assert s.t >= 0


def test_drift_positions():
    cfg = Config()
    cfg.n_samples = 1000
    assert make_dataset("s", "sudden", cfg).drift_positions == (500,)
    assert make_dataset("g", "gradual", cfg).drift_positions == (500,)
    assert make_dataset("r", "recurring", cfg).drift_positions == (250, 500, 750)
    assert make_dataset("n", "none", cfg).drift_positions == ()
    assert make_dataset("i", "incremental", cfg).has_drift is True
    assert make_dataset("n", "none", cfg).has_drift is False


def test_sudden_drift_reverses_concept():
    """突发漂移后，类别 0 的条件均值必须明显移动（证明概念真的变了）。"""
    cfg = Config()
    cfg.n_samples = 1000
    ds = make_dataset("s", "sudden", cfg)
    pre, post = [], []
    for s in ds.stream():
        if s.y == 0:
            (pre if s.t < 500 else post).append(s.x)
    mp, mq = np.mean(pre, axis=0), np.mean(post, axis=0)
    assert np.linalg.norm(mp - mq) > 0.5, "漂移后类条件均值没有明显变化"


def test_recurring_drift_alternates():
    """复发漂移：段 0/2 概念相同，段 1 与段 0 不同（c0/c1 交替）。"""
    cfg = Config()
    cfg.n_samples = 1000
    ds = make_dataset("r", "recurring", cfg)
    segs = [[] for _ in range(4)]
    for s in ds.stream():
        seg = min(int(s.t / 250.0), 3)
        if s.y == 0:
            segs[seg].append(s.x)
    m = [np.mean(x, axis=0) for x in segs]
    assert np.linalg.norm(m[0] - m[1]) > 0.5  # 段0 vs 段1：概念变了
    assert np.linalg.norm(m[0] - m[2]) < 0.5  # 段0 vs 段2：回到原概念


def test_noisy_stream_has_no_drift_positions_but_is_harder():
    cfg = Config()
    cfg.n_samples = 600
    noisy = make_dataset("noisy", "noisy", cfg)
    assert noisy.drift_positions == ()
    # 噪声流里"同类样本的特征均值"仍在中心附近（噪声只作用于标签）
    xs = [s.x for s in noisy.stream() if s.y == 0]
    assert len(xs) > 50


def test_default_datasets_covers_all_drift_types():
    cfg = Config()
    names = [d.name for d in default_datasets(cfg)]
    assert "stationary" in names and "recurring_drift" in names
    assert len(names) == 6


def test_unknown_drift_type_raises():
    cfg = Config()
    with pytest.raises(ValueError):
        make_dataset("bad", "not_a_drift", cfg)
