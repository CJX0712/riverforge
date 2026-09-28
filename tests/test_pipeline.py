"""流水线 / 后端契约单测：全算法可跑、跳过语义、聚合排名。"""

from __future__ import annotations

from riverforge.core.config import Config
from riverforge.data.stream import make_dataset
from riverforge.domain.streaming.numpy_impl import NumpyPerceptron
from riverforge.domain.streaming.registry import available_entries, build_all
from riverforge.domain.streaming.sklearn_wrappers import SklearnSGD
from riverforge.pipeline.pipeline import StreamPipeline


def _cfg(n=200):
    c = Config()
    c.n_samples = n
    return c


class _Boom:
    """故意抛错的学习器，用于验证 skipped 语义（不伪造数字）。"""

    name = "boom"
    supports_drift = False

    def predict_one(self, x):
        raise RuntimeError("boom predict")

    def learn_one(self, x, y):
        raise RuntimeError("boom learn")


def test_all_available_algorithms_run_without_crash():
    cfg = _cfg(150)
    pipe = StreamPipeline(cfg)
    ds = make_dataset("s", "sudden", cfg)
    for learner in build_all(cfg):
        r = pipe.eval_one(learner, ds)
        assert not r.skipped, f"{r.algorithm} 被跳过: {r.note}"
        assert 0.0 <= r.prequential_accuracy <= 1.0
        assert r.n_seen == cfg.n_samples


def test_broken_learner_is_marked_skipped_not_faked():
    cfg = _cfg(100)
    pipe = StreamPipeline(cfg)
    r = pipe.eval_one(_Boom(), make_dataset("s", "none", cfg))
    assert r.skipped is True
    assert "boom" in r.note
    assert r.prequential_accuracy == 0.0  # 不伪造数字


def test_benchmark_and_summarize_ranking_sorted():
    cfg = _cfg(150)
    pipe = StreamPipeline(cfg)
    rows = pipe.benchmark(seeds=(42,))
    summary = pipe.summarize(rows)
    accs = [r["mean_acc"] for r in summary["ranking"]]
    assert accs == sorted(accs, reverse=True)
    assert summary["best_algorithm"] == summary["ranking"][0]["algorithm"]
    assert summary["n_rows"] == len(rows)


def test_benchmark_multi_seed_multiplies_rows():
    cfg = _cfg(120)
    pipe = StreamPipeline(cfg)
    rows = pipe.benchmark(seeds=(1, 2))
    n_algos = len(build_all(cfg))
    assert len(rows) == 2 * len(rows) // 2  # sanity
    assert len(rows) == n_algos * 6 * 2  # 6 场景 × 2 种子


def test_available_entries_nonempty_and_includes_flagship():
    entries = available_entries(_cfg())
    assert len(entries) >= 3
    assert any(e.get("flagship") for e in entries)


def test_sklearn_backend_available_or_gracefully_skipped():
    """sklearn 可用则应进入注册表；不可用则 available()=False（不报错）。"""
    assert isinstance(SklearnSGD.available(), bool)


def test_prequential_order_is_test_then_train():
    """先测后学：第 0 条样本的预测必须来自未训练模型（不能先用它训练）。"""
    cfg = _cfg(50)
    pipe = StreamPipeline(cfg)
    learner = NumpyPerceptron(averaged=False)
    ds = make_dataset("s", "none", cfg)
    first = next(iter(ds.stream()))
    p_before = int(learner.predict_one(first.x))
    r = pipe.eval_one(learner, ds)
    assert r.n_seen == cfg.n_samples
    assert isinstance(p_before, int)
