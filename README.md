# RiverForge

> 世界顶级在线学习（流式 / 概念漂移）系统 · 作者：晨星 (CJX0712)

RiverForge 复用顶级开源（**scikit-learn** 增量学习 / **river** 流式 ML / **numpy** 离线兜底），
在**零下载可跑**的前提下提供从纯 numpy 在线学习器到漂移感知集成的完整流水线，
评测口径严格 **prequential（先测后学）**，并内置**确定性可复现基准**与 **63 项数值不变量单测**。

## 核心特性

- **13 算法 / 4 档分层**：纯 numpy 兜底（感知机 / PA-I / 在线逻辑回归）→
  scikit-learn `partial_fit`（SGD / 在线线性 SVM / PA）→ river 流式 SOTA
  （LogisticRegression / HoeffdingTree / GaussianNB / ADWINBagging / SRP，缺失自动跳过）→ 旗舰。
- **旗舰 DriftForge**：漂移感知在线加权集成 ——
  基池可信度（EWMA）加权投票 + ADWIN/DDM 漂移探测 + 漂移即重置（权重回均匀 + 基重置）。
- **6 档难度梯度流式场景**：平稳 → 突发（类中心反转）→ 渐进 → 增量旋转 → 复发漂移 → 高噪声。
- **漂移探测器**：river ADWIN（理论误报率界）+ 纯 numpy DDM（Laplace 平滑修正版，零依赖兜底）。
- **确定性可复现**：固定 `random_state=42` 与种子组 (42/123/7)，重跑逐位一致，落盘 `benchmark.json`。
- **63 项数值不变量自测**：PA-I 间隔恰好=1、DDM 平稳零误报（含退化 bug 回归测试）、
  numpy ↔ river 指标交叉验证逐位一致、跳过语义（不伪造数字）。

## 安装

```bash
pip install -r requirements.txt   # numpy / scipy / scikit-learn / river
pip install pytest ruff           # 可选：开发
```

## 快速开始

```python
from riverforge.core.config import Config
from riverforge.data.stream import make_dataset
from riverforge.pipeline.pipeline import StreamPipeline
from riverforge.domain.streaming.ensemble import DriftForge

cfg = Config()
pipe = StreamPipeline(cfg)
ds = make_dataset("sudden_drift", "sudden", cfg)
res = pipe.eval_one(DriftForge(cfg=cfg), ds)
print(res.prequential_accuracy, res.drift_recall)  # 精度 + 漂移召回
```

命令行 / 端到端演示：

```bash
riverforge benchmark            # 3 种子 × 6 场景 × 13 算法 -> 打印排名
riverforge run --scenario sudden_drift --drift-type sudden
riverforge datasets             # 列出 6 档流式场景
riverforge algorithms           # 列出可用算法（不可用后端自动跳过）
python -m riverforge.examples.run_demo   # 落盘 benchmark.json
```

## 基准结果（3 种子 × 6 场景 × 13 算法 = 234 评测，v0.1.0）

| 排名 | 算法 | mean prequential acc | min acc | mean AUC |
|----:|------|--------:|--------:|--------:|
| 1 | **drift-forge（旗舰）** | **0.828** | 0.645 | 0.848 |
| 2 | reset-perceptron（消融：单模型+漂移重置） | 0.825 | 0.660 | 0.876 |
| 3 | numpy-online-logistic | 0.818 | 0.652 | 0.870 |
| 4 | river-logistic | 0.774 | 0.634 | 0.816 |
| 5 | sklearn-svm-hinge | 0.772 | 0.556 | 0.832 |
| … | sklearn-sgd / numpy-pa / sklearn-pa | 0.75–0.77 | — | — |
| 10 | river-adwin-bagging | 0.688 | 0.489 | 0.759 |
| 12 | numpy-perceptron（静态，无漂移处理） | 0.683 | 0.512 | 0.738 |
| 13 | river-gaussian-nb | 0.680 | 0.506 | 0.748 |

**关键结论（消融，详见 docs/architecture.md §4）**
- 漂移处理的增益远大于集成：静态感知机 0.683 → +漂移重置 0.825（**+14.2pt**）是最大单一增益；
- 旗舰 DriftForge 在全部算法中排名第一（0.828），较最强单基 +1.0pt；
- 漂移召回：sudden / gradual / recurring = **1.00**，平稳流误报 0.00（incremental 为连续旋转、
  无突变点，两种探测器均不报警——属诚实行为而非缺陷）。

## 架构

```
riverforge/
├── core/                 Config / StreamSample / Dataset / StreamResult / 错误码 / 契约接口
├── data/stream.py        6 档难度梯度流式场景（确定性，可复现）
├── domain/streaming/
│   ├── numpy_impl.py     纯 numpy 离线兜底（感知机 / PA-I / 在线逻辑回归）
│   ├── sklearn_wrappers.py   partial_fit 增量学习（SGD / SVM / PA）
│   ├── optional.py       river 流式 SOTA（自动探测，缺失跳过）
│   ├── detectors.py      NumpyDDM + ADWIN(river)
│   ├── ensemble.py       旗舰 DriftForge + 消融 DriftResetWrapper
│   └── registry.py       4 档分层注册表
├── eval/metrics.py       prequential accuracy / kappa / rank-AUC / 漂移召回
├── pipeline/pipeline.py  benchmark + summarize（聚合排名）
└── examples/run_demo.py  端到端演示
```

详见 [docs/architecture.md](docs/architecture.md)。

## 作者

晨星 (CJX0712) — 随机创新世界顶级 AI 系统系列之 RiverForge。

## 许可

MIT
