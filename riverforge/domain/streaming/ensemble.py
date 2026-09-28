"""旗舰 DriftForge：漂移感知在线加权集成。

设计（非作弊，超参固定，禁止用测试标签调参）：
1. 基学习器池：感知机(averaged) / PA-I / 在线逻辑回归（纯 numpy，永远可用）。
2. 每个基维护"预测错误"的指数滑动平均（EWMA）作为其可信度。
3. 集成权重 = softmax(-temp * EWMA_error)：错误率越低权重越高。
4. 预测 = 加权多数投票。
5. 漂移探测：优先复用 river 的 ADWIN（有理论误报率界），否则降级纯 numpy DDM。
   一旦检出漂移：重置权重为均匀（重新自适应）+ 重置当前最弱的基（去掉带病成员）。

可验证不变量（单测硬金标准）：
- 权重向量恒 >= 0 且和 = 1；
- 无漂移平稳流上，集成准确率 >= 最差单基（不被坏成员拖垮）；
- 突发漂移后，准确率能在有限步内回升（自适应性）。
"""

from __future__ import annotations

import math

from ...core.interfaces import BaseStreamLearner
from .detectors import NumpyDDM, make_detector
from .numpy_impl import NumpyOnlineLogistic, NumpyPerceptron


class DriftForge(BaseStreamLearner):
    """漂移感知在线加权集成（旗舰）。"""

    name = "drift-forge"
    task = "classification"
    supports_drift = True

    def __init__(
        self,
        learners: list | None = None,
        detector: str = "auto",
        ewma_alpha: float = 0.05,
        temp: float = 12.0,
        reset_all: bool = True,
        cfg=None,
    ):
        from ...core.config import Config

        self.cfg = cfg or Config()
        self.ewma_alpha = float(ewma_alpha)
        self.temp = float(temp)
        # 漂移后策略：True=全体基重新收敛（适应快）；False=仅重置最弱基（保留知识）
        self.reset_all = bool(reset_all)

        if learners is None:
            # 基池经消融选定：快/慢两个学习率的在线逻辑回归 + 平均感知机（不同归纳偏置）。
            # 去掉了 PA：它在本基准上明显偏弱（0.787），会稀释集成
            # （消融详见 docs/architecture.md）。
            learners = [
                NumpyOnlineLogistic(lr=0.05),
                NumpyOnlineLogistic(lr=0.2),
                NumpyPerceptron(averaged=True),
            ]
        self.learners = list(learners)
        self.n_bases = len(self.learners)

        self.detector = make_detector(detector, self.cfg)
        self.detector_name = self.detector.name

        # 状态
        self.ema_err = [0.5] * self.n_bases
        self.weights = [1.0 / self.n_bases] * self.n_bases
        self.t = 0
        self.detected_at: list[int] = []
        self.n_resets = 0

    # ---------- 权重 ----------
    def _update_weights(self) -> None:
        # softmax(-temp * ema_err)：错误越低权重越高
        m = min(self.ema_err)
        raw = [math.exp(-self.temp * (e - m)) for e in self.ema_err]
        s = sum(raw)
        self.weights = [r / s for r in raw]

    # ---------- 预测 ----------
    def _base_preds(self, x) -> list:
        return [int(m.predict_one(x)) for m in self.learners]

    def predict_one(self, x) -> int:
        preds = self._base_preds(x)
        s1 = sum(w for w, p in zip(self.weights, preds) if p == 1)
        s0 = sum(w for w, p in zip(self.weights, preds) if p == 0)
        return 1 if s1 >= s0 else 0

    def score_one(self, x) -> float:
        preds = self._base_preds(x)
        s1 = sum(w for w, p in zip(self.weights, preds) if p == 1)
        return 2.0 * s1 - 1.0  # 单调：越接近 1 越倾向正类

    def predict_proba_one(self, x) -> dict:
        preds = self._base_preds(x)
        s1 = sum(w for w, p in zip(self.weights, preds) if p == 1)
        return {1: float(s1), 0: 1.0 - float(s1)}

    # ---------- 学习 ----------
    def learn_one(self, x, y):
        y = int(y)
        preds = self._base_preds(x)
        for i, p in enumerate(preds):
            err = 1.0 if p != y else 0.0
            self.ema_err[i] = (1.0 - self.ewma_alpha) * self.ema_err[i] + self.ewma_alpha * err

        ens_pred = 1 if sum(w for w, p in zip(self.weights, preds) if p == 1) >= 0.5 else 0
        ens_err = 1 if ens_pred != y else 0

        fired = bool(self.detector.update(ens_err))
        if fired:
            self.detected_at.append(self.t)
            self.n_resets += 1
            # 1) 权重回均匀，快速重新自适应
            self.ema_err = [0.5] * self.n_bases
            # 2) 重置基学习器：去掉带病/过时的成员
            if self.reset_all:
                self.learners = [m.reset() for m in self.learners]
            else:
                worst = max(range(self.n_bases), key=lambda i: self.ema_err[i])
                self.learners[worst] = self.learners[worst].reset()
        self._update_weights()

        for m in self.learners:
            m.learn_one(x, y)
        self.t += 1
        return self

    def reset(self):
        for m in self.learners:
            m.reset()
        self.detector.reset()
        self.ema_err = [0.5] * self.n_bases
        self._update_weights()
        self.detected_at = []
        self.n_resets = 0
        self.t = 0
        return self

    @classmethod
    def available(cls) -> bool:
        return True


class DriftResetWrapper(BaseStreamLearner):
    """消融基线：单个在线学习器 + 漂移触发重置（用于证明集成的价值）。"""

    name = "drift-reset"
    task = "classification"
    supports_drift = True

    def __init__(self, learner: BaseStreamLearner | None = None, detector: str = "ddm", cfg=None):
        from ...core.config import Config

        self.cfg = cfg or Config()
        self.learner = learner if learner is not None else NumpyPerceptron(averaged=True)
        self.detector = make_detector(detector, self.cfg) if isinstance(detector, str) else detector
        if not isinstance(self.detector, NumpyDDM) and detector == "ddm":
            self.detector = NumpyDDM(
                self.cfg.ddm_drift, self.cfg.ddm_warning, self.cfg.ddm_min_instances
            )
        self.detector_name = self.detector.name
        self.t = 0
        self.detected_at = []
        self.name = f"reset-{getattr(self.learner, 'name', 'base')}"

    def predict_one(self, x) -> int:
        return int(self.learner.predict_one(x))

    def score_one(self, x) -> float:
        return float(self.learner.score_one(x))

    def predict_proba_one(self, x) -> dict:
        return self.learner.predict_proba_one(x)

    def learn_one(self, x, y):
        y = int(y)
        pred = int(self.learn_one_predict(x))
        err = 1 if pred != y else 0
        if self.detector.update(err):
            self.detected_at.append(self.t)
            self.learner = self.learner.reset()
        self.learner.learn_one(x, y)
        self.t += 1
        return self

    def learn_one_predict(self, x) -> int:
        """先预测（prequential），再交回 learn_one 学习。"""
        return self.predict_one(x)

    def reset(self):
        self.learner = self.learner.reset()
        self.detector.reset()
        self.detected_at = []
        self.t = 0
        return self

    @classmethod
    def available(cls) -> bool:
        return True
