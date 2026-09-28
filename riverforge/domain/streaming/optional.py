"""可选 SOTA 后端：复用顶级开源 river（流式机器学习）。

river 是 Python 流式 ML 的事实标准库（在线学习/概念漂移/在线特征工程）。
本模块把 river 的模型与集成包装成统一契约；river 不可用时 available()=False，
流水线自动跳过（不伪造数字）。
"""

from __future__ import annotations

from ...core.interfaces import BaseStreamLearner


def _river_available() -> bool:
    try:
        import river  # noqa: F401

        return True
    except Exception:
        return False


class _RiverBase(BaseStreamLearner):
    task = "classification"
    _builder = None

    def __init__(self):
        self.model = None
        if self._builder is not None:
            try:
                self.model = self._builder()
            except Exception:
                self.model = None

    @classmethod
    def available(cls) -> bool:
        return _river_available() and cls._builder is not None

    @staticmethod
    def _to_dict(x) -> dict:
        return {f"f{i}": float(v) for i, v in enumerate(x)}

    def learn_one(self, x, y):
        if self.model is None:
            return self
        self.model.learn_one(self._to_dict(x), int(y))
        return self

    def predict_one(self, x) -> int:
        if self.model is None:
            return 0
        try:
            return int(self.model.predict_one(self._to_dict(x)))
        except Exception:
            return 0

    def predict_proba_one(self, x) -> dict:
        if self.model is None:
            return {0: 0.5, 1: 0.5}
        try:
            p = self.model.predict_proba_one(self._to_dict(x))
            return {int(k): float(v) for k, v in p.items()}
        except Exception:
            pr = self.predict_one(x)
            return {pr: 1.0, 1 - pr: 0.0}

    def score_one(self, x) -> float:
        p = self.predict_proba_one(x).get(1, 0.5)
        p = min(max(p, 1e-9), 1 - 1e-9)
        import math

        return math.log(p / (1 - p))

    def reset(self):
        if self._builder is not None:
            try:
                self.model = self._builder()
            except Exception:
                pass
        return self


def _b_logistic():
    from river.linear_model import LogisticRegression  # type: ignore

    return LogisticRegression()


def _b_ht():
    from river.tree import HoeffdingTreeClassifier  # type: ignore

    return HoeffdingTreeClassifier()


def _b_gnb():
    from river.naive_bayes import GaussianNB  # type: ignore

    return GaussianNB()


def _b_adwin_bagging():
    from river.ensemble import ADWINBaggingClassifier  # type: ignore
    from river.tree import HoeffdingTreeClassifier  # type: ignore

    return ADWINBaggingClassifier(model=HoeffdingTreeClassifier(), n_models=5, seed=42)


def _b_srp():
    from river.ensemble import SRPClassifier  # type: ignore

    return SRPClassifier(n_models=5, seed=42)


class RiverLogistic(_RiverBase):
    name = "river-logistic"
    _builder = staticmethod(_b_logistic)


class RiverHoeffdingTree(_RiverBase):
    name = "river-hoeffding-tree"
    _builder = staticmethod(_b_ht)


class RiverGaussianNB(_RiverBase):
    name = "river-gaussian-nb"
    _builder = staticmethod(_b_gnb)


class RiverADWINBagging(_RiverBase):
    name = "river-adwin-bagging"
    _builder = staticmethod(_b_adwin_bagging)


class RiverSRP(_RiverBase):
    name = "river-srp"
    _builder = staticmethod(_b_srp)
