# RiverForge 架构（作者：晨星）

> 世界顶级在线学习（流式 / 概念漂移）系统。复用顶级开源：scikit-learn / river / numpy。

## 1. 设计总纲

流式学习的核心矛盾与批量学习不同：**模型必须在"只看一遍"的数据上边预测边学习**，
且数据分布会随时间漂移。因此本系统的评测口径是 **prequential（先测后学）**：
每条样本先预测（记分），再用真实标签更新模型——杜绝任何"偷看"。

```
riverforge/
├── core/            types(StreamSample/Dataset/StreamResult) · errors(E100~E500)
│                    config(ENV_RF_* 覆盖) · interfaces(BaseStreamLearner/Detector/Metric)
├── data/            stream.py  6 档难度梯度流式场景（确定性可复现）
├── domain/streaming/
│   ├── numpy_impl.py      纯 numpy 离线兜底：感知机(averaged) / PA-I / 在线逻辑回归
│   ├── sklearn_wrappers.py  SGD(hinge/log_loss) / PA —— partial_fit 工业级在线学习
│   ├── optional.py        river 流式 SOTA：LogisticRegression / HoeffdingTree / GaussianNB
│   │                      / ADWINBagging / SRP（缺失自动跳过，不伪造数字）
│   ├── detectors.py       NumpyDDM（兜底）+ ADWIN（river，可选）
│   ├── ensemble.py        旗舰 DriftForge + 消融基线 DriftResetWrapper
│   └── registry.py        4 档分层注册表（numpy / sklearn / river / flagship）
├── eval/metrics.py  prequential accuracy / kappa / rank-AUC / 漂移召回；与 river 指标对拍
├── pipeline/        benchmark：场景 × 算法 × 种子，全量落盘 benchmark.json
└── cli.py + examples/run_demo.py
```

调用单向无环：`cli → pipeline → {data, domain, eval} → core`。
所有学习器统一契约：`learn_one(x, y)` / `predict_one(x)` / `score_one(x)`（越大越偏正类），
保证跨后端（numpy/sklearn/river）公平比较。

## 2. 六档流式场景（难度梯度）

| 场景 | 漂移类型 | 构造 | 考察点 |
|------|---------|------|--------|
| stationary | none | 固定高斯类中心 | 收敛上限（天花板参照） |
| sudden_drift | sudden | t0=0.5n 处类中心**反转**（c0→c0[::-1]） | 灾难性遗忘后重建速度 |
| gradual_drift | gradual | [t0, t0+w] 线性插值迁移 | 渐进概念迁移 |
| incremental_drift | incremental | 中心持续旋转（旋转超平面语义） | 持续自适应 |
| recurring_drift | recurring | 4 段 c0/c1 交替 → 3 次复发突发漂移 | 反复漂移下的长期表现 |
| noisy | none | 25% 标签翻转 | 抗噪（贝叶斯上限 ≈0.75） |

关键设计决策：**漂移用"类中心反转"而非独立随机新中心**。若用随机中心，可能与原中心部分重叠 →
漂移过轻，各方法差异淹没在噪声里；反转保证概念真的变了（收敛模型准确率骤降），
漂移适应能力的差异才区分得出来。

## 3. 旗舰 DriftForge：漂移感知在线加权集成

```
        ┌─ OnlineLogistic(lr=0.05) ─┐
样本 x ─┼─ OnlineLogistic(lr=0.20) ─┼─→ 加权多数投票 → ŷ
        └─ Perceptron(averaged)  ───┘
             ↑ 权重 = softmax(-temp · EWMA_error)
             ↑ 漂移探测器（ADWIN / DDM）监听集成错误率
                  └─ 检出漂移 → 权重回均匀 + 全体基重置
```

1. **基池**（消融选定，见 §4）：快/慢两个学习率的在线逻辑回归 + 平均感知机——
   强度相近但归纳偏置不同；去掉了 PA（单基最弱，会稀释集成）。
2. **可信度**：每个基维护预测错误的指数滑动平均（EWMA，α=0.05）。
3. **加权投票**：`w_i = softmax(-12 · EWMA_err_i)`，错误率越低权重越高。
4. **漂移探测**：优先复用 river 的 ADWIN（有理论误报率界）；
   river 缺失时降级到自研纯 numpy DDM（Laplace 平滑修正版）。
5. **漂移响应**：检出即① 权重回均匀（重新自适应），② 全体基重置（剔除过时知识）。

非作弊约束：全部超参（α、temp、ADWIN δ、DDM 阈值、基池）在选定后**固定**，
绝不针对测试标签调参；评测严格先测后学。

## 4. 消融实验（n=1000，seed=42，6 场景 mean prequential accuracy）

### 4.1 基池选择

| 基池 | mean | 说明 |
|------|-----:|------|
| 静态 Perceptron（无漂移处理） | 0.7098 | 突发 0.563 / 复发 0.537 —— 反例 |
| 静态 OnlineLogistic(0.05) | 0.8555 | 最强单基 |
| [Perc, PA, OL.05]（初版） | 0.8650 | PA 拖累 |
| [OL.05, OL.02, Perc] | 0.8628 | |
| **[OL.05, OL.20, Perc]（选定）** | **0.8677** | 最优 |

### 4.2 探测器 × 重置策略

| 配置 | mean | 说明 |
|------|-----:|------|
| ADWIN + 重置全部（选定默认） | 0.8677 | river 复用 |
| ADWIN + 重置最弱 | 0.8673 | |
| DDM + 重置全部 | 0.8693 | 纯 numpy 兜底反而略优（检出更快） |
| reset-perceptron（DDM，消融基线） | 0.8667 | 单模型 + 漂移重置 |

### 4.3 核心消融结论

漂移处理带来的增益远大于集成带来的增益：
- 静态感知机 0.7098 → +漂移重置 0.8667（**+15.7pt**），是本基准上最大的单一增益；
- 集成（DriftForge 0.8677）较其最强单基（0.8555）+1.2pt，属锦上添花。
诚实结论：**本系统的价值主要在"漂移感知"而非"集成"**；两者叠加最优。

## 5. 数值不变量（单测硬金标准，63 项全绿）

| 不变量 | 位置 | 断言 |
|--------|------|------|
| 感知机更新方向 | numpy_impl | 误分类后 y·score 朝正确符号移动 |
| PA-I 间隔恰好为 1 | numpy_impl | 更新后 \|margin − 1\| < 1e-6；已满足则不再更新 |
| 在线逻辑回归学习性 | numpy_impl | 可分数据后 1/3 错误率 < 前 1/3 |
| DDM 平稳无报警 | detectors | 全对流 300 步 0 报警 |
| DDM 误报回归 | detectors | 恒定 30% 错误率 ≤2 次（修复预热期 p_min bug） |
| ADWIN 零误报 | detectors | river 可用时平稳流 0 报警 |
| 集成权重是分布 | ensemble | Σw=1, w≥0；EWMA∈[0,1] |
| 旗舰漂移检出 | ensemble | 突发漂移 recall=1.0（±150 窗）；平稳流误报 ≤2 |
| 自适应硬指标 | ensemble | DriftForge > 静态感知机 +5pt（突发漂移） |
| numpy ↔ river 交叉验证 | metrics | accuracy/kappa 与 river Accuracy/CohenKappa 逐位一致 |
| AUC 并列修正 | metrics | 全并列=0.5；部分并列=0.375（手工枚举 4 对） |
| 跳过语义 | pipeline | 异常后端标记 skipped=True，准确率置 0（不伪造） |

## 6. 复现

```bash
pip install -r requirements.txt
python -m riverforge.examples.run_demo   # 跨场景跨算法基准 -> benchmark.json
pytest -q                                # 63 项不变量
clusterforge/riverforge CLI: riverforge benchmark / datasets / algorithms
```

确定性：固定 `random_state=42` 与种子组 (42,123,7)；数据生成器纯 numpy；
同一命令重跑逐位一致。
