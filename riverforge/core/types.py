"""核心数据类型：流样本、数据集、评测结果、行。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass


@dataclass
class StreamSample:
    """单条流式样本。x 为特征向量，y 为标签/目标，t 为到达序号。"""

    x: list
    y: object
    t: int


@dataclass
class Dataset:
    """一个流式数据集（带真实标签、真实概念漂移类型、可复现生成器）。"""

    name: str
    task: str  # 'classification' | 'regression'
    n_features: int
    n_classes: int
    drift_type: str  # 'none' | 'sudden' | 'gradual' | 'incremental' | 'noisy'
    generator: Callable[[], Iterator[StreamSample]]
    n_samples: int = 2000
    has_drift: bool = False
    drift_positions: tuple = ()
    note: str = ""

    def stream(self) -> Iterator[StreamSample]:
        return self.generator()


@dataclass
class StreamResult:
    """单算法 × 单场景的评测结果。"""

    algorithm: str
    scenario: str
    task: str
    prequential_accuracy: float
    kappa: float
    auc: float
    mae: float
    rmse: float
    drift_recall: float
    drift_false_alarm: float
    n_seen: int
    runtime_s: float
    supports_drift: bool
    skipped: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "task": self.task,
            "prequential_accuracy": round(self.prequential_accuracy, 4),
            "kappa": round(self.kappa, 4),
            "auc": round(self.auc, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "drift_recall": round(self.drift_recall, 4),
            "drift_false_alarm": round(self.drift_false_alarm, 4),
            "n_seen": self.n_seen,
            "runtime_s": round(self.runtime_s, 4),
            "supports_drift": self.supports_drift,
            "skipped": self.skipped,
            "note": self.note,
        }


@dataclass
class EvalRow:
    """与 StreamResult 兼容的轻量行（避免重复）。"""

    algorithm: str
    scenario: str
    task: str
    prequential_accuracy: float = 0.0
    kappa: float = 0.0
    auc: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    drift_recall: float = 0.0
    drift_false_alarm: float = 0.0
    n_seen: int = 0
    runtime_s: float = 0.0
    supports_drift: bool = False
    skipped: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "task": self.task,
            "prequential_accuracy": round(self.prequential_accuracy, 4),
            "kappa": round(self.kappa, 4),
            "auc": round(self.auc, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "drift_recall": round(self.drift_recall, 4),
            "drift_false_alarm": round(self.drift_false_alarm, 4),
            "n_seen": self.n_seen,
            "runtime_s": round(self.runtime_s, 4),
            "supports_drift": self.supports_drift,
            "skipped": self.skipped,
            "note": self.note,
        }
