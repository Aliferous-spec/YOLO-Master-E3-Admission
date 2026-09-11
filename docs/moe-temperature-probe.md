# MoE 温度探针：`anneal_mixture_temperatures(factor=2.0)` 对离散路由指标的影响

目的：把「上游公开 API `anneal_mixture_temperatures` 是否真的会改变离散 routing 指标」
固化成一次性、可复现、可追溯的只读探针——不重训、不 sweep、不改 forward、不改模型、
不改 routing schema、不改上游 YOLO-Master、不改动任何既有 evidence。

结论预告：**API 生效且影响可观测，但效应有限**，因此不继续做完整 intervention（见 §5）。

## 1. 范围（必须与性能结论区分）

- 观测对象是**未训练随机初始化 checkpoint** 的 **routing 结构量**（离散 top-k 选择派生出的
  分布统计），不是训练后的路由形态，也不是精度/损失/训练性能结论。
- 「无变化」在本文档中**只**表示该指标在该次观测下未变，不解释为性能结论。
- 本次只改温度一个变量，不涉及训练、不涉及 loss 权重、不涉及 expert 数量或 top-k。

## 2. 复现方式

```bat
set PYTHONPATH=D:\YOLO-Master
python -m scripts.probe_moe_temperature ^
  --baseline-root D:\YOLO-Master ^
  --out artifacts/temperature/moe-temp-factor2-20260912/moe_temperature_probe.json
```

- `PYTHONPATH` 用于把 `ultralytics` 解析到部署基线 checkout（与
  `docs/p1-training-slowdown-result.md` §1 同一手法）。脚本内置 import gate：若
  `ultralytics` 实际解析到别的 checkout（例如 editable 安装的 review 目录），直接报错而不是
  静默跑错环境。
- 默认 `factor=2.0`、`min_temp=0.3`；这两个常量被
  `tests/test_probe_moe_temperature.py` 钉住，改动会让测试失败。

## 3. 公平性约束（两臂只差 temperature）

| 约束 | 做法 |
|---|---|
| 同一 checkpoint | 只构建一次 `YOLO(...)`，两臂不重载权重；artifact 记录构建时的 `checkpoint.weights_sha256` |
| 同一输入 | 同一张 `torch.randn(1,3,640,640)` 张量被四个 arm 复用 |
| 同一环境 | 同一解释器/同一 `ultralytics`（见 artifact `environment`） |
| 同一采集配置 | 同一 collector `scripts.routing_capture.capture_records`（`e3-routing/v1`，`training=True`），沿用既有 MoE routing case（`DetailAwareLowRankHybridAdaptiveGateMoE`，`yolo-master-n.yaml`） |
| 隔离内置调度 | 第一臂之前先 `configure_mixture_temperature_schedule(module, external=True)`，否则上游 per-forward 余弦调度会自己漂移温度，两臂就不再只差温度 |
| 隔离 BatchNorm | train 模式 forward 会改写 BN running stats，故先 `snapshot_bn_running_state`，每个 arm 前 `restore_bn_running_state` |
| RNG 可区分 | 每臂跑两次（`*_repeat`），非零 delta 不能归因于 RNG |

温度是**升**（1.2 → 2.4）：`factor=2.0` 是上游默认退火方向（0.97，降温）的反向探针。

## 4. 结果

### 4.1 API 调用

`anneal_mixture_temperatures(module, factor=2.0, min_temp=0.3)` 返回 `updated_modules: 3`，
三个 MoE router 的温度实测从 `1.200000` 变为 `2.400000`：

| router | before | after |
|---|---|---|
| `model.5.routing` | 1.200000 | 2.400000 |
| `model.8.routing` | 1.200000 | 2.400000 |
| `model.11.routing` | 1.200000 | 2.400000 |

API 同时置 `_external_temperature_schedule=True`；这正是让 `AdaptiveGateMoE._update_temperature()`
在后续 train-mode forward 中提前返回、不再覆盖退火值的原因（实测温度在四个 arm 间保持 2.4）。

### 4.2 逐层 before / after 与 absolute delta

| layer | E | 指标 | baseline | factor=2.0 | delta |
|---|---|---|---|---|---|
| `model.5` | 4 | top1_share | 1.000000 | 1.000000 | +0.000000 |
| `model.5` | 4 | entropy_nats | 0.000000 | 0.000000 | +0.000000 |
| `model.5` | 4 | entropy_normalized | 0.000000 | 0.000000 | +0.000000 |
| `model.5` | 4 | gini | 0.750000 | 0.750000 | +0.000000 |
| `model.5` | 4 | dominant_expert | 3 | 3 | 未变 |
| `model.8` | 8 | top1_share | 0.536595 | 0.518322 | -0.018273 |
| `model.8` | 8 | entropy_nats | 0.690466 | 0.692476 | +0.002010 |
| `model.8` | 8 | entropy_normalized | 0.332044 | 0.333010 | +0.000966 |
| `model.8` | 8 | gini | 0.759149 | 0.754580 | -0.004569 |
| `model.8` | 8 | dominant_expert | 1 | 1 | 未变 |
| `model.11` | 16 | top1_share | 0.541292 | 0.520669 | -0.020623 |
| `model.11` | 16 | entropy_nats | 0.689733 | 0.692292 | +0.002559 |
| `model.11` | 16 | entropy_normalized | 0.248769 | 0.249692 | +0.000923 |
| `model.11` | 16 | gini | 0.880162 | 0.877584 | -0.002578 |
| `model.11` | 16 | dominant_expert | 0 | 0 | 未变 |

`model.5` 的 `mean_router_probs` 确实变了（`[0.259502, 0.226428, 0.249975, 0.264095]` →
`[0.254817, 0.238025, 0.250096, 0.257062]`），但没有跨越 top-2 边界，因此离散指标逐位不变——
这正是「软概率动了、离散选择没动」的情形，不是测量失败。完整 usage 向量见 artifact
`layers[*].expert_usage`。

### 4.3 聚合 delta（3 个 MoE layer）

| 指标 | 变化层数 | mean delta |
|---|---|---|
| top1_share | 2 / 3 | -0.012965 |
| entropy_nats | 2 / 3 | +0.001523 |
| entropy_normalized | 2 / 3 | +0.000630 |
| gini | 2 / 3 | -0.002382 |
| dominant_expert | 0 / 3 | 未变 |

方向自洽：温度升高 → softmax 更平 → top1_share 下降、熵微升、Gini 微降。

### 4.4 Deterministic 重跑

同一温度下重复 arm 的结果**逐位一致**，两臂各自的 `sha256` 也一致：

| arm | sha256（层级结果行） | 与同温度重跑一致 |
|---|---|---|
| baseline | `22b4d39b4d685b602a87cb1bfb95ee60f9c93e2b678b530748b972f94595e377` | 是 |
| baseline（重跑） | `22b4d39b4d685b602a87cb1bfb95ee60f9c93e2b678b530748b972f94595e377` | 是 |
| factor=2.0 | `4daf5e95c2c9f3f84a39c0897101af322393b7294579da555bc4923820dd43f3` | 是 |
| factor=2.0（重跑） | `4daf5e95c2c9f3f84a39c0897101af322393b7294579da555bc4923820dd43f3` | 是 |

因此 §4.2 / §4.3 的非零 delta 不是 RNG 漂移，是温度造成的。该探针另经独立一次进程级重跑核对，
指标逐位相同。

## 5. 结论

1. **API 生效**：`anneal_mixture_temperatures(factor=2.0)` 确实更新了 3 个 MoE router 的
   temperature（1.2 → 2.4），并正确接管调度权（冻结内置余弦调度），温度在两臂之间保持稳定。
2. **对离散 routing 指标可观测**：3 个 MoE layer 中 2 个的 top1_share / entropy /
   normalized entropy / Gini 在同 checkpoint、同输入、同采集配置下发生变化，且可确定性复现。
3. **效应有限**：top1_share 变化约 2 个百分点、Gini 变化 < 0.005、归一化熵变化 < 0.001；
   3 个 layer 的 dominant expert 全部未变；`model.5` 的离散指标完全未动（其路由本身已坍缩到
   单专家，top-2 边界无从跨越）。
4. **因此不继续做完整 intervention**：在未训练 checkpoint 上，温度不是能重新分配专家选择的
   杠杆；`docs/p0-three-seed-evidence.md` 已记录该检查点下 MoE 路由本就 one-hot 坍缩，
   继续放大温度只会继续压缩 logit 间距，不会改变已坍缩的 argmax 结构。
5. 若后续确需评估「温度对路由均衡」的干预效果，应先有可用的训练后 checkpoint，并把
   判据（效应量阈值、目标 layer）一并预注册；本次结果不构成该判据。

## 6. artifact

- 路径：`artifacts/temperature/moe-temp-factor2-20260912/moe_temperature_probe.json`
- sha256：`a4d6794f4d6b53b22aec81f4ede2cdf7a7d4ccfe32732342bf2856872efd2f8d`
- schema：`e3-temperature-probe/v1`（仅本探针使用，不修改 `e3-routing/v1` 或任何既有协议）
- 环境（artifact `environment` 自采）：Windows-10-10.0.26200-SP0 / Python 3.11.9 /
  torch 2.13.0+cpu / ultralytics 8.4.101；`captured_at` 2026-09-12T01:42:07+08:00
- 基线：`D:\YOLO-Master` @ `aa5d2e20c109b96f4a0c68f667ed2694586ef745`；checkpoint
  `random_init_from_yaml`，`weights_sha256`
  `0809638cd04866194aa4e5ab0ac631b93a9ce894d2b5c8d72bac0fa8be4911ac`（四臂共用同一权重集）
- 本次只新增该一个结果文件，未修改任何既有 evidence。

## 7. 非目标

- 不重训、不 sweep、不改 `factor` / `min_temp` 之外任何参数、不做消融矩阵。
- 不修改上游 YOLO-Master、不改 forward、不改 `e3-routing/v1` schema、不改既有协议与 evidence。
- 不声称训练性能改善（本探针全程未训练、未评估精度）。