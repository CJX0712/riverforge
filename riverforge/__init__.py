"""RiverForge — 世界顶级在线学习（流式/概念漂移）系统。

作者: 晨星 (CJX0712)

复用顶级开源: scikit-learn(partial_fit 增量学习) / river(流式 ML + ADWIN) / numpy(离线兜底)。
旗舰 DriftForge: 漂移感知在线加权集成
（漂移探测 + 指数滑动权重 + 漂移时重置）。零下载可跑。
"""

__version__ = "0.1.0"
__author__ = "晨星"

from .core.config import Config
from .core.types import Dataset, EvalRow, StreamResult, StreamSample

__all__ = [
    "Config",
    "Dataset",
    "StreamResult",
    "StreamSample",
    "EvalRow",
    "__version__",
    "__author__",
]
