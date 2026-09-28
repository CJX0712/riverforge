"""流式数据生成器（带真实标签 + 真实概念漂移，确定性可复现）。

设计目标（难度梯度，避免天花板效应）：
- stationary：平稳高斯混合，任何增量模型都能收敛到高准确率。
- sudden：t0 时刻类中心整体互换（标签反转），已收敛模型准确率骤降 → 检验漂移适应速度。
- gradual：在 [t0, t0+w] 窗口内线性插值迁移中心 → 渐进漂移。
- incremental：中心持续旋转（旋转超平面语义），概念连续变化 → 检验持续自适应。
- noisy：平稳但 25% 标签翻转 → 贝叶斯上限压低，检验抗噪（所有方法 ceiling 一致）。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..core.types import Dataset, StreamSample

DRIFT_TYPES = ("none", "sudden", "gradual", "incremental", "recurring", "noisy")


def _unit_vectors(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    V = rng.normal(size=(n, d))
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(norms, 1e-12)


def gaussian_centers(
    n_classes: int, n_features: int, rng: np.random.Generator, separation: float
) -> np.ndarray:
    """把 n_classes 个中心放到半径 separation 的随机方向上。"""
    return _unit_vectors(n_classes, n_features, rng) * separation


def _rotation(d: int, theta: float) -> np.ndarray:
    """在前两个维度上旋转 theta（d<2 时为恒等）。"""
    R = np.eye(d)
    if d >= 2:
        c, s = np.cos(theta), np.sin(theta)
        R[0, 0], R[0, 1] = c, -s
        R[1, 0], R[1, 1] = s, c
    return R


def _make_stream(
    drift_type: str,
    seed: int,
    n_samples: int,
    n_features: int,
    n_classes: int,
    separation: float,
    label_noise: float,
    pos_frac: float,
    width_frac: float,
) -> tuple[Iterator[StreamSample], list]:
    if drift_type not in DRIFT_TYPES:
        raise ValueError(f"unknown drift_type: {drift_type}")

    rng = np.random.default_rng(seed)
    c0 = gaussian_centers(n_classes, n_features, rng, separation)
    # 漂移后的中心 = 漂移前中心"反转"（类别↔中心互换），
    # 保证概念变化足够剧烈：收敛模型的准确率会跌到接近 0，必须真正重新学习。
    # （若用独立随机中心，可能与 c0 部分重叠 → 漂移过轻，漂移适应能力区分不出来）
    c1 = c0[::-1].copy()

    t0 = int(n_samples * pos_frac)
    w = max(1, int(n_samples * width_frac))
    omega = 0.006

    if drift_type == "sudden":
        drift_positions = [t0]
    elif drift_type == "gradual":
        drift_positions = [t0]
    elif drift_type == "incremental":
        drift_positions = [int(n_samples * 0.25), int(n_samples * 0.5), int(n_samples * 0.75)]
    elif drift_type == "recurring":
        drift_positions = [
            int(n_samples * 0.25),
            int(n_samples * 0.5),
            int(n_samples * 0.75),
        ]
    else:
        drift_positions = []

    def gen() -> Iterator[StreamSample]:
        r = np.random.default_rng(seed + 1000)
        for t in range(n_samples):
            if drift_type in ("none", "noisy"):
                C = c0
            elif drift_type == "sudden":
                C = c0 if t < t0 else c1
            elif drift_type == "gradual":
                a = 0.0 if t < t0 else min(1.0, (t - t0) / w)
                C = (1.0 - a) * c0 + a * c1
            elif drift_type == "recurring":
                # 4 段，交替 c0 / c1 -> 3 次突发复发漂移
                seg = int(t / max(n_samples / 4.0, 1e-9))
                C = c0 if seg % 2 == 0 else c1
            elif drift_type == "incremental":
                C = c0 @ _rotation(n_features, omega * t).T
            else:  # pragma: no cover
                C = c0

            cls = int(r.integers(0, n_classes))
            x = C[cls] + r.normal(0.0, 0.85, size=n_features)
            y = cls

            noise_p = 0.25 if drift_type == "noisy" else label_noise
            if noise_p > 0 and r.random() < noise_p:
                # 翻到"另一个"类（保证不等于原类）
                k = int(r.integers(0, max(1, n_classes - 1)))
                y = (cls + 1 + k) % n_classes

            yield StreamSample(x=[float(v) for v in x], y=int(y), t=t)

    return gen(), drift_positions


def make_dataset(
    name: str,
    drift_type: str,
    cfg=None,
    seed: int | None = None,
) -> Dataset:
    """按配置构造一个流式数据集。"""
    if cfg is None:
        from ..core.config import Config

        cfg = Config()
    sd = cfg.random_state if seed is None else seed

    params = dict(
        drift_type=drift_type,
        seed=sd,
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_classes=cfg.n_classes,
        separation=cfg.separation,
        label_noise=cfg.label_noise,
        pos_frac=cfg.drift_position_frac,
        width_frac=cfg.drift_width_frac,
    )

    # 工厂：每次调用 stream() 都新建生成器（避免复用已耗尽的生成器）
    def factory():
        g, _ = _make_stream(**params)
        return g

    _, positions = _make_stream(**params)

    return Dataset(
        name=name,
        task="classification",
        n_features=cfg.n_features,
        n_classes=cfg.n_classes,
        drift_type=drift_type,
        generator=factory,
        n_samples=cfg.n_samples,
        has_drift=drift_type in ("sudden", "gradual", "incremental", "recurring"),
        drift_positions=tuple(positions),
        note=f"seed={sd}, n={cfg.n_samples}, d={cfg.n_features}, k={cfg.n_classes}",
    )


def default_datasets(cfg=None) -> list:
    """5 档难度梯度：平稳 → 突发 → 渐进 → 增量 → 噪声。"""
    return [
        make_dataset("stationary", "none", cfg),
        make_dataset("sudden_drift", "sudden", cfg),
        make_dataset("gradual_drift", "gradual", cfg),
        make_dataset("incremental_drift", "incremental", cfg),
        make_dataset("recurring_drift", "recurring", cfg),
        make_dataset("noisy", "noisy", cfg),
    ]


def list_datasets(cfg=None) -> list:
    return [d.name for d in default_datasets(cfg)]
