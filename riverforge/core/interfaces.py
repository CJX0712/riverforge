"""接口契约（Protocol）：所有学习器/探测器/指标统一语义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStreamLearner(ABC):
    """在线学习器：一次一条样本 learn_one / predict_one。

    语义约定：
    - 分类标签统一为 int（0..n_classes-1）。
    - predict_one 返回 int 标签；score_one 返回"越接近正类越大"的分数。
    - available() 为 False 时流水线自动跳过（不伪造数字）。
    """

    name: str = "base"
    task: str = "classification"
    supports_drift: bool = False

    @abstractmethod
    def learn_one(self, x, y) -> BaseStreamLearner:
        """用一条样本增量更新模型。"""

    @abstractmethod
    def predict_one(self, x) -> Any:
        """预测一条样本的标签。"""

    def predict_proba_one(self, x) -> dict:
        """返回 {label: prob}；默认由 score 构造二分类概率。"""
        s = self.score_one(x)
        p = 1.0 / (1.0 + _exp(-s))
        return {1: p, 0: 1.0 - p}

    def score_one(self, x) -> float:
        """决策分数（分类=离超平面的有符号距离语义）。"""
        return float(self.predict_one(x))

    def reset(self) -> BaseStreamLearner:
        """漂移后重置内部状态（默认子类实现返回自身）。"""
        return self

    @classmethod
    def available(cls) -> bool:
        return True

    def clone(self) -> BaseStreamLearner:
        return type(self)(**getattr(self, "_ctor_kwargs", {}))


def _exp(v: float) -> float:
    import math

    v = max(-60.0, min(60.0, v))
    return math.exp(v)


class BaseDriftDetector(ABC):
    """概念漂移探测器：喂入 0/1 错误指示，返回是否检出漂移。"""

    name: str = "base-detector"
    drift_detected: bool = False

    @abstractmethod
    def update(self, error: int) -> bool:
        """喂入本步是否出错(1=错,0=对)，返回 True 表示检出漂移。"""

    @abstractmethod
    def reset(self) -> None:
        """重置统计。"""

    @classmethod
    def available(cls) -> bool:
        return True


class BaseMetric(ABC):
    """流式指标：update 后 get。"""

    name: str = "base-metric"

    @abstractmethod
    def update(self, y_true, y_pred) -> BaseMetric: ...

    @abstractmethod
    def get(self) -> float: ...
