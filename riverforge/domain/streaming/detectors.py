"""概念漂移探测器。

- NumpyDDM：纯 numpy 的 DDM（Drift Detection Method），零依赖兜底。
  维护错误率的均值 p 与标准差 s；当 p + s >= p_min + drift_level * s_min 判漂移，
  p + s >= p_min + warning_level * s_min 判预警（Gama et al. 2004）。
- ADWINDetector：复用顶级开源 river 的 ADWIN（滑动窗口自适应，理论保证的误报率界），
  river 不可用时 available()=False，流水线自动降级到 NumpyDDM。
"""

from __future__ import annotations

import math

from ...core.interfaces import BaseDriftDetector


class NumpyDDM(BaseDriftDetector):
    """DDM：基于错误率均值/标准差的漂移检测（纯 numpy，无外部依赖）。"""

    name = "ddm"

    def __init__(
        self, drift_level: float = 3.0, warning_level: float = 2.0, min_instances: int = 40
    ):
        self.drift_level = float(drift_level)
        self.warning_level = float(warning_level)
        self.min_instances = int(min_instances)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.n_err = 0.0
        self.p = 1.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.drift_detected = False
        self.warning_zone = False
        self.n_detections = 0

    def update(self, error: int) -> bool:
        err = 1.0 if error else 0.0
        self.n += 1
        self.n_err += err
        # Laplace 平滑：p、s 恒 > 0。
        # 否则"全对"时 p=s=0 -> threshold=0 -> 无条件误报（cur=0 >= 0），
        # 这是个真实的退化 bug：第 min_instances 步就会假报警。
        p = (self.n_err + 1.0) / (self.n + 2.0)
        s = math.sqrt(p * (1.0 - p) / (self.n + 2.0))
        self.p, self.s = p, s

        self.drift_detected = False
        self.warning_zone = False

        # 冷启动保护：预热期内不更新 p_min/s_min，
        # 否则会拿"前几个碰巧猜对"的低错误率当基线，导致每 min_instances 步误报一次。
        if self.n < self.min_instances:
            return False

        if self.s < self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        threshold = self.p_min + self.drift_level * self.s_min
        warn_at = self.p_min + self.warning_level * self.s_min
        cur = self.p + self.s

        if cur > threshold:
            self.drift_detected = True
            saved = self.n_detections + 1
            self.reset()
            self.n_detections = saved  # reset 会清零，漂移后补偿保留累计计数
            return True
        if cur >= warn_at:
            self.warning_zone = True
        return False

    def clone(self) -> NumpyDDM:
        return NumpyDDM(self.drift_level, self.warning_level, self.min_instances)


class _ADWINBase:
    """延迟导入 river.drift.ADWIN。"""

    @staticmethod
    def import_adwin():
        try:
            from river.drift import ADWIN  # type: ignore

            return ADWIN
        except Exception:
            return None


class ADWINDetector(BaseDriftDetector):
    """复用 river 的 ADWIN（自适应滑动窗口）。river 缺失时 available()=False。"""

    name = "adwin"

    def __init__(self, delta: float = 0.002):
        self.delta = float(delta)
        self._cls = _ADWINBase.import_adwin()
        self._det = None
        if self._cls is not None:
            self._det = self._cls(delta=self.delta)
        self.drift_detected = False
        self.n_detections = 0

    @classmethod
    def available(cls) -> bool:
        return _ADWINBase.import_adwin() is not None

    def reset(self) -> None:
        if self._det is None:
            return
        # river ADWIN 无显式 reset，重建
        self._det = self._cls(delta=self.delta)
        self.drift_detected = False

    def update(self, error: int) -> bool:
        if self._det is None:
            return False
        self._det.update(1 if error else 0)
        self.drift_detected = bool(getattr(self._det, "drift_detected", False))
        if self.drift_detected:
            self.n_detections += 1
        return self.drift_detected


def make_detector(name: str = "auto", cfg=None) -> BaseDriftDetector:
    """工厂：auto 优先 river ADWIN，不可用时降级 numpy DDM。"""
    from ...core.config import Config

    cfg = cfg or Config()
    key = (name or "auto").lower()
    if key in ("auto", "adwin"):
        if cfg.use_river and ADWINDetector.available():
            return ADWINDetector(delta=cfg.adwin_delta)
        if key == "adwin":
            raise RuntimeError("ADWIN 请求但 river 不可用")
        return NumpyDDM(cfg.ddm_drift, cfg.ddm_warning, cfg.ddm_min_instances)
    if key == "ddm":
        return NumpyDDM(cfg.ddm_drift, cfg.ddm_warning, cfg.ddm_min_instances)
    raise ValueError(f"unknown detector: {name}")
