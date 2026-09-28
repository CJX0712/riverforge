"""基准流水线：场景 × 算法 × 种子的 prequential 交错评测。"""

from __future__ import annotations

import time

from ..core.config import Config
from ..core.types import StreamResult
from ..data.stream import default_datasets
from ..domain.streaming.registry import build_all
from ..eval.metrics import PrequentialEvaluator, drift_scores


class StreamPipeline:
    """prequential（先测后学）基准编排。"""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.from_env()

    # ---------- 单算法 × 单场景 ----------
    def eval_one(self, learner, dataset, window: int | None = None) -> StreamResult:
        window = window if window is not None else self.cfg.drift_window
        ev = PrequentialEvaluator()
        n = 0
        t0 = time.perf_counter()
        try:
            for s in dataset.stream():
                try:
                    pred = int(learner.predict_one(s.x))
                except Exception:
                    pred = 0
                try:
                    score = float(learner.score_one(s.x))
                except Exception:
                    score = float(pred == 1)
                ev.update(s.y, pred, score)
                learner.learn_one(s.x, s.y)
                n += 1
        except Exception as e:  # 后端不可用/异常 -> 标记 skipped，不伪造数字
            return StreamResult(
                algorithm=getattr(learner, "name", "?"),
                scenario=dataset.name,
                task=dataset.task,
                prequential_accuracy=0.0,
                kappa=0.0,
                auc=0.5,
                mae=0.0,
                rmse=0.0,
                drift_recall=0.0,
                drift_false_alarm=0.0,
                n_seen=n,
                runtime_s=time.perf_counter() - t0,
                supports_drift=getattr(learner, "supports_drift", False),
                skipped=True,
                note=f"{type(e).__name__}: {str(e)[:80]}",
            )

        m = ev.get()
        elapsed = time.perf_counter() - t0
        detected = getattr(learner, "detected_at", [])
        recall, fa = drift_scores(detected, dataset.drift_positions, n, window=window)
        return StreamResult(
            algorithm=getattr(learner, "name", "?"),
            scenario=dataset.name,
            task=dataset.task,
            prequential_accuracy=m["accuracy"],
            kappa=m["kappa"],
            auc=m["auc"],
            mae=0.0,
            rmse=0.0,
            drift_recall=recall,
            drift_false_alarm=fa,
            n_seen=n,
            runtime_s=elapsed,
            supports_drift=getattr(learner, "supports_drift", False),
            skipped=False,
            note="",
        )

    # ---------- 全量基准 ----------
    def benchmark(self, seeds=None, datasets=None, algos=None):
        cfg = self.cfg
        seeds = seeds or cfg.benchmark_seeds
        rows = []

        for sd in seeds:
            cfg_s = Config.from_env()
            cfg_s.random_state = int(sd)
            for k, v in cfg.as_dict().items():
                if k not in ("random_state", "benchmark_seeds"):
                    setattr(cfg_s, k, v)

            ds_list = datasets if datasets is not None else default_datasets(cfg_s)
            algo_list = algos if algos is not None else build_all(cfg_s)

            for ds in ds_list:
                # 同一场景对每个算法都要一份全新流（生成器工厂保证）
                for factory in algo_list:
                    learner = _fresh(factory, cfg_s)
                    rows.append(self.eval_one(learner, ds))
        return rows

    # ---------- 聚合 ----------
    def summarize(self, rows) -> dict:
        by_algo = {}
        for r in rows:
            if r.skipped:
                continue
            d = by_algo.setdefault(
                r.algorithm, {"acc": [], "auc": [], "kappa": [], "sec": [], "n": 0}
            )
            d["acc"].append(r.prequential_accuracy)
            d["auc"].append(r.auc)
            d["kappa"].append(r.kappa)
            d["sec"].append(r.runtime_s)
            d["n"] += 1

        ranking = []
        for name, d in by_algo.items():
            ranking.append(
                {
                    "algorithm": name,
                    "mean_acc": sum(d["acc"]) / len(d["acc"]),
                    "min_acc": min(d["acc"]),
                    "mean_auc": sum(d["auc"]) / len(d["auc"]),
                    "mean_kappa": sum(d["kappa"]) / len(d["kappa"]),
                    "mean_runtime_s": sum(d["sec"]) / len(d["sec"]),
                    "n_runs": d["n"],
                }
            )
        ranking.sort(key=lambda x: x["mean_acc"], reverse=True)

        return {
            "best_algorithm": ranking[0]["algorithm"] if ranking else None,
            "ranking": ranking,
            "n_rows": len(rows),
            "n_skipped": sum(1 for r in rows if r.skipped),
        }


def _fresh(proto, cfg):
    """从原型生成一个干净实例（同类型 + 同构造器参数）。"""
    try:
        if type(proto) is type(proto) and hasattr(proto, "_ctor_kwargs"):
            return type(proto)(**proto._ctor_kwargs)
    except Exception:
        pass
    try:
        if hasattr(proto, "clone"):
            return proto.clone()
    except Exception:
        pass
    # 无法干净复制时，按类型重建
    cls = type(proto)
    try:
        if cls.__name__ == "DriftForge":
            return cls(cfg=cfg)
        if cls.__name__ == "DriftResetWrapper":
            from ..domain.streaming.numpy_impl import NumpyPerceptron

            return cls(learner=NumpyPerceptron(averaged=True), cfg=cfg)
        return cls()
    except Exception:
        return proto
