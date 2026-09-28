"""配置：支持 ENV_RF_* 环境变量覆盖。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v not in (None, "") else default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v in (None, ""):
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # 可复现性
    random_state: int = 42
    benchmark_seeds: tuple = (42, 123, 7)

    # 数据流
    n_samples: int = 1000
    n_features: int = 6
    n_classes: int = 2
    separation: float = 1.6
    label_noise: float = 0.05
    drift_position_frac: float = 0.5
    drift_width_frac: float = 0.2

    # 漂移探测（numpy DDM 兜底）
    ddm_warning: float = 2.0
    ddm_drift: float = 3.0
    ddm_min_instances: int = 40

    # 漂移探测（river ADWIN，可选）
    adwin_delta: float = 0.002

    # 旗舰 DriftForge（固定超参，禁止用测试标签调参）
    ewma_alpha: float = 0.05
    softmax_temp: float = 12.0

    # 漂移召回容差：检出点落在真实漂移点 ±drift_window 内算命中
    # （漂移探测本身有延迟，100 步太紧会把 107 步的 ADWIN 正确报警判成误报）
    drift_window: int = 150

    # 后端开关
    use_river: bool = True
    use_sklearn: bool = True

    def as_dict(self) -> dict:
        d = asdict(self)
        d["benchmark_seeds"] = list(self.benchmark_seeds)
        return d

    @classmethod
    def from_env(cls) -> Config:
        seeds = os.environ.get("ENV_RF_SEEDS")
        bench = tuple(int(s) for s in seeds.split(",")) if seeds else cls.benchmark_seeds
        return cls(
            random_state=_env_int("ENV_RF_SEED", cls.random_state),
            benchmark_seeds=bench,
            n_samples=_env_int("ENV_RF_N_SAMPLES", cls.n_samples),
            n_features=_env_int("ENV_RF_N_FEATURES", cls.n_features),
            n_classes=_env_int("ENV_RF_N_CLASSES", cls.n_classes),
            separation=_env_float("ENV_RF_SEPARATION", cls.separation),
            label_noise=_env_float("ENV_RF_LABEL_NOISE", cls.label_noise),
            drift_position_frac=_env_float("ENV_RF_DRIFT_POS", cls.drift_position_frac),
            drift_width_frac=_env_float("ENV_RF_DRIFT_WIDTH", cls.drift_width_frac),
            ddm_warning=_env_float("ENV_RF_DDM_WARN", cls.ddm_warning),
            ddm_drift=_env_float("ENV_RF_DDM_DRIFT", cls.ddm_drift),
            ddm_min_instances=_env_int("ENV_RF_DDM_MIN", cls.ddm_min_instances),
            adwin_delta=_env_float("ENV_RF_ADWIN_DELTA", cls.adwin_delta),
            ewma_alpha=_env_float("ENV_RF_EWMA", cls.ewma_alpha),
            softmax_temp=_env_float("ENV_RF_TEMP", cls.softmax_temp),
            drift_window=_env_int("ENV_RF_DRIFT_WINDOW", cls.drift_window),
            use_river=_env_bool("ENV_RF_USE_RIVER", cls.use_river),
            use_sklearn=_env_bool("ENV_RF_USE_SKLEARN", cls.use_sklearn),
        )
