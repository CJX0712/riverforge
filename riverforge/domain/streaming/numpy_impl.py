"""纯 numpy 在线学习器（零下载离线兜底）。

- NumpyPerceptron：感知机（可选 averaged，Freund & Schapire 1999 投票感知机）。
- NumpyPassiveAggressive：PA-I（Crammer et al. 2006），仅在间隔不足时更新。
- NumpyOnlineLogistic：在线逻辑回归（SGD + 稳定 sigmoid）。

不变量（单测硬金标准）：
- 感知机：误分类时权重沿 y·x 方向移动（决策分数朝正确方向单调增加）。
- PA：更新后该样本间隔 margin >= 1 - 1e-6（进入正确侧且满足软间隔）。
- 在线逻辑回归：似然/损失在可分数据上随步数下降（EMA 单调不增趋势）。
"""

from __future__ import annotations

import numpy as np

from ...core.interfaces import BaseStreamLearner


def _as_vec(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def _bin_sign(y) -> float:
    return 1.0 if int(y) >= 1 else -1.0


class NumpyPerceptron(BaseStreamLearner):
    """感知机（二分类，标签 {0,1}），支持 averaged 投票。"""

    name = "numpy-perceptron"
    task = "classification"

    def __init__(self, lr: float = 0.1, averaged: bool = True):
        self.lr = float(lr)
        self.averaged = bool(averaged)
        self._ctor_kwargs = {"lr": lr, "averaged": averaged}
        self.w = None
        self.b = 0.0
        self._w_sum = None
        self._b_sum = 0.0
        self._n_updates = 0

    def _ensure(self, d: int) -> None:
        if self.w is None or self.w.shape[0] != d:
            self.w = np.zeros(d)
            self.b = 0.0
            self._w_sum = np.zeros(d)
            self._b_sum = 0.0
            self._n_updates = 0

    def _weights(self):
        if self.averaged and self._n_updates > 0:
            return self._w_sum / self._n_updates, self._b_sum / self._n_updates
        return self.w, self.b

    def score_one(self, x) -> float:
        xv = _as_vec(x)
        self._ensure(xv.shape[0])
        w, b = self._weights()
        return float(np.dot(w, xv) + b)

    def learn_one(self, x, y):
        xv = _as_vec(x)
        self._ensure(xv.shape[0])
        s = _bin_sign(y)
        pred = float(np.dot(self.w, xv) + self.b)
        if s * pred <= 0.0:  # 误分类才更新
            self.w = self.w + self.lr * s * xv
            self.b += self.lr * s
        # 平均感知机：每一步都累积（含未更新的步）
        self._w_sum = self._w_sum + self.w
        self._b_sum += self.b
        self._n_updates += 1
        return self

    def predict_one(self, x) -> int:
        return 1 if self.score_one(x) >= 0.0 else 0

    def reset(self):
        self.w = None
        self.b = 0.0
        self._w_sum = None
        self._b_sum = 0.0
        self._n_updates = 0
        return self


class NumpyPA(BaseStreamLearner):
    """PA-I：loss = max(0, 1 - y(w·x+b))，tau = loss/(||x||²+1)。"""

    name = "numpy-pa"
    task = "classification"

    def __init__(self, aggressiveness: float = 1.0):
        self.C = float(aggressiveness)
        self._ctor_kwargs = {"aggressiveness": aggressiveness}
        self.w = None
        self.b = 0.0

    def _ensure(self, d: int) -> None:
        if self.w is None or self.w.shape[0] != d:
            self.w = np.zeros(d)
            self.b = 0.0

    def score_one(self, x) -> float:
        xv = _as_vec(x)
        self._ensure(xv.shape[0])
        return float(np.dot(self.w, xv) + self.b)

    def learn_one(self, x, y):
        xv = _as_vec(x)
        self._ensure(xv.shape[0])
        s = _bin_sign(y)
        margin = s * (float(np.dot(self.w, xv)) + self.b)
        loss = max(0.0, 1.0 - margin)
        if loss > 0.0:
            norm2 = float(np.dot(xv, xv))
            tau = (
                loss / (norm2 + 1.0 / max(self.C, 1e-12)) if self.C != 1.0 else loss / (norm2 + 1.0)
            )
            self.w = self.w + tau * s * xv
            self.b += tau * s
        return self

    def predict_one(self, x) -> int:
        return 1 if self.score_one(x) >= 0.0 else 0

    def reset(self):
        self.w = None
        self.b = 0.0
        return self


class NumpyOnlineLogistic(BaseStreamLearner):
    """在线逻辑回归（SGD），稳定 sigmoid 防溢出。"""

    name = "numpy-online-logistic"
    task = "classification"

    def __init__(self, lr: float = 0.05, l2: float = 1e-4):
        self.lr = float(lr)
        self.l2 = float(l2)
        self._ctor_kwargs = {"lr": lr, "l2": l2}
        self.w = None
        self.b = 0.0

    def _ensure(self, d: int) -> None:
        if self.w is None or self.w.shape[0] != d:
            self.w = np.zeros(d)
            self.b = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + float(np.exp(-min(z, 60.0))))
        e = float(np.exp(max(z, -60.0)))
        return e / (1.0 + e)

    def proba_one(self, x) -> float:
        xv = _as_vec(x)
        self._ensure(xv.shape[0])
        return self._sigmoid(float(np.dot(self.w, xv)) + self.b)

    def score_one(self, x) -> float:
        p = self.proba_one(x)
        # logit，保持与 predict 一致的单调性
        p = min(max(p, 1e-9), 1 - 1e-9)
        return float(np.log(p / (1.0 - p)))

    def learn_one(self, x, y):
        xv = _as_vec(x)
        self._ensure(xv.shape[0])
        yv = 1.0 if int(y) >= 1 else 0.0
        p = self.proba_one(xv)
        g = p - yv
        self.w = self.w - self.lr * (g * xv + self.l2 * self.w)
        self.b -= self.lr * g
        return self

    def predict_one(self, x) -> int:
        return 1 if self.proba_one(x) >= 0.5 else 0

    def predict_proba_one(self, x) -> dict:
        p = self.proba_one(x)
        return {1: p, 0: 1.0 - p}

    def reset(self):
        self.w = None
        self.b = 0.0
        return self
