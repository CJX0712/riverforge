"""算法注册表：4 档分层（numpy 兜底 / sklearn / river 可选 / 旗舰+消融）。"""

from __future__ import annotations

from .ensemble import DriftForge, DriftResetWrapper
from .numpy_impl import NumpyOnlineLogistic, NumpyPA, NumpyPerceptron
from .optional import (
    RiverADWINBagging,
    RiverGaussianNB,
    RiverHoeffdingTree,
    RiverLogistic,
    RiverSRP,
)
from .sklearn_wrappers import SklearnPA, SklearnSGD, SklearnSVM

REGISTRY = [
    # ---- Tier 0: 纯 numpy 离线兜底（零下载可跑）----
    {"cls": NumpyPerceptron, "tier": "numpy", "label": "numpy-perceptron"},
    {"cls": NumpyPA, "tier": "numpy", "label": "numpy-pa"},
    {"cls": NumpyOnlineLogistic, "tier": "numpy", "label": "numpy-online-logistic"},
    # ---- Tier 1: scikit-learn partial_fit（工业级在线学习）----
    {"cls": SklearnSGD, "tier": "sklearn", "label": "sklearn-sgd"},
    {"cls": SklearnSVM, "tier": "sklearn", "label": "sklearn-svm-hinge"},
    {"cls": SklearnPA, "tier": "sklearn", "label": "sklearn-pa"},
    # ---- Tier 2: river 流式 SOTA（可选，缺失自动跳过）----
    {"cls": RiverLogistic, "tier": "river", "label": "river-logistic"},
    {"cls": RiverHoeffdingTree, "tier": "river", "label": "river-hoeffding-tree"},
    {"cls": RiverGaussianNB, "tier": "river", "label": "river-gaussian-nb"},
    {"cls": RiverADWINBagging, "tier": "river", "label": "river-adwin-bagging"},
    {"cls": RiverSRP, "tier": "river", "label": "river-srp"},
    # ---- Tier 3: 消融 + 旗舰 ----
    {"cls": DriftResetWrapper, "tier": "ablation", "label": "reset-perceptron"},
    {"cls": DriftForge, "tier": "flagship", "flagship": True, "label": "drift-forge"},
]


def available_entries(cfg=None) -> list:
    """返回当前环境可用的算法条目（不可用后端自动跳过）。"""
    out = []
    for e in REGISTRY:
        cls = e["cls"]
        try:
            if not cls.available():
                continue
        except Exception:
            continue
        out.append(e)
    return out


def build_all(cfg=None) -> list:
    """实例化全部可用算法。"""
    from ...core.config import Config

    cfg = cfg or Config()
    learners = []
    for e in available_entries(cfg):
        cls = e["cls"]
        try:
            if cls is DriftForge:
                learners.append(DriftForge(cfg=cfg))
            elif cls is DriftResetWrapper:
                learners.append(DriftResetWrapper(learner=NumpyPerceptron(averaged=True), cfg=cfg))
            else:
                learners.append(cls())
        except Exception:
            continue
    return learners


def list_algorithms(cfg=None) -> list:
    return [(m.name, getattr(m, "supports_drift", False)) for m in build_all(cfg)]
