"""复用顶级开源 scikit-learn 的增量学习器（partial_fit）。

SGDClassifier / PassiveAggressiveClassifier 的 partial_fit 是工业级在线学习的事实标准；
这里包装成 learn_one / predict_one 语义，与纯 numpy 兜底对齐，保证跨后端公平比较。

注意：估计器类通过 `_factory` 静态工厂惰性导入（类级别），
使得 available() 能在 import 失败时正确返回 False（而不是实例属性残留导致误判）。
"""

from __future__ import annotations

import numpy as np

from ...core.interfaces import BaseStreamLearner


class _SklearnPartialFit(BaseStreamLearner):
    """partial_fit 系列公共逻辑。"""

    task = "classification"
    name = "sklearn-base"
    _factory = None  # staticmethod -> estimator class（子类覆盖）

    def __init__(self, classes=(0, 1), **kwargs):
        self.classes = np.array(list(classes))
        self._kwargs = dict(kwargs)
        self._ctor_kwargs = dict(kwargs)
        self._fitted = False
        est = self._get_estimator_cls()
        self.model = est(**kwargs) if est is not None else None

    @classmethod
    def _get_estimator_cls(cls):
        if cls._factory is None:
            return None
        try:
            return cls._factory()
        except Exception:
            return None

    @classmethod
    def available(cls) -> bool:
        return cls._get_estimator_cls() is not None

    def _to2d(self, x) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(1, -1)

    def learn_one(self, x, y):
        if self.model is None:
            return self
        X = self._to2d(x)
        yv = np.array([int(y)])
        if not self._fitted:
            self.model.partial_fit(X, yv, classes=self.classes)
            self._fitted = True
        else:
            self.model.partial_fit(X, yv)
        return self

    def predict_one(self, x) -> int:
        if self.model is None or not self._fitted:
            return 0
        return int(self.model.predict(self._to2d(x))[0])

    def predict_proba_one(self, x) -> dict:
        if self.model is None or not self._fitted:
            return {0: 0.5, 1: 0.5}
        try:
            p = self.model.predict_proba(self._to2d(x))[0]
            return {int(c): float(pi) for c, pi in zip(self.model.classes_, p)}
        except Exception:
            pr = int(self.predict_one(x))
            return {pr: 1.0, 1 - pr: 0.0}

    def score_one(self, x) -> float:
        if self.model is None or not self._fitted:
            return 0.0
        try:
            return float(self.model.decision_function(self._to2d(x))[0])
        except Exception:
            p = self.predict_proba_one(x).get(1, 0.5)
            p = min(max(p, 1e-9), 1 - 1e-9)
            return float(np.log(p / (1 - p)))

    def reset(self):
        est = self._get_estimator_cls()
        self.model = est(**self._kwargs) if est is not None else None
        self._fitted = False
        return self


def _f_sgd_log():
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier


def _f_sgd_hinge():
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier


def _f_pa():
    from sklearn.linear_model import PassiveAggressiveClassifier

    return PassiveAggressiveClassifier


class SklearnSGD(_SklearnPartialFit):
    """SGDClassifier(log_loss) 增量版 —— 在线逻辑回归基线。"""

    name = "sklearn-sgd"
    _factory = staticmethod(_f_sgd_log)

    def __init__(self, loss: str = "log_loss", alpha: float = 1e-4, classes=(0, 1)):
        super().__init__(classes=classes, loss=loss, alpha=alpha, random_state=42)


class SklearnSVM(_SklearnPartialFit):
    """SGDClassifier(hinge) —— 在线线性 SVM（间隔视角基线）。"""

    name = "sklearn-svm-hinge"
    _factory = staticmethod(_f_sgd_hinge)

    def __init__(self, alpha: float = 1e-4, classes=(0, 1)):
        super().__init__(classes=classes, loss="hinge", alpha=alpha, random_state=42)


class SklearnPA(_SklearnPartialFit):
    """PassiveAggressiveClassifier 增量版 —— 与 numpy PA 对拍。"""

    name = "sklearn-pa"
    _factory = staticmethod(_f_pa)

    def __init__(self, C: float = 1.0, classes=(0, 1)):
        super().__init__(classes=classes, C=C, random_state=42)
