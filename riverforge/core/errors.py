"""RiverForge 错误码体系（E100~E500）。"""

from __future__ import annotations


class RiverForgeError(Exception):
    """所有 RiverForge 错误的基类。"""

    code = "E000"

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class DataError(RiverForgeError):
    """数据/流相关错误 (E100)。"""

    code = "E100"


class ConfigError(RiverForgeError):
    """配置错误 (E200)。"""

    code = "E200"


class BackendUnavailable(RiverForgeError):
    """可选 SOTA 后端不可用 (E300)。"""

    code = "E300"


class NumericalError(RiverForgeError):
    """数值不变量被破坏 (E400)。"""

    code = "E400"


class PipelineError(RiverForgeError):
    """流水线/评测错误 (E500)。"""

    code = "E500"
