# P0 三 Seed 证据（2026-09-09）

任务书要求 P0 证据覆盖至少 3 个随机种子。本页记录三个 seed 的 smoke 运行、
manifest 校验结果与跨 seed 指标对比，作为 P0 红线补齐的凭证。

## 运行记录

| seed | run_id | smoke | manifest 校验 |
|---|---|---|---|
| 0 | `seed0-20260909-232515-5dbb` | PASS | PASS |
| 1 | `seed1-20260909-232807-2076` | PASS | PASS |
| 2 | `seed2-20260909-232807-616f` | PASS | PASS |

- 批量驱动：`python -m scripts.run_smoke_seeds --baseline-root D:/YOLO-Master --seeds 0,1,2`
- 逐 run 复核：`python -m scripts.run_e3_smoke --verify-artifacts artifacts/smoke/<run_id>`
- 环境：验收基线 venv（Python 3.11.9 / torch 2.13.0+cpu / ultralytics 8.4.101，
  见各 run 的 `environment.json`）；基线 `D:\YOLO-Master` @ `aa5d2e20c109b96f4a0c68f667ed2694586ef745`
  （与运行期 venv editable 安装的 review checkout 是两个不同 checkout，不可视为同一个 commit）。

## 跨 Seed 指标对比（canonical `routing_records.jsonl`，15 模块/运行）

| family | module | entropy_norm（3 seed） | gini（3 seed） |
|---|---|---|---|
| latent | model.23/24/25 | 1.0000 / 1.0000 / 1.0000 | 0.0000 ×3 |
| moe | model.5 | −0.0000 ×3 | 0.7500 ×3 |
| moe | model.8 | −0.0000 ×3 | 0.8750 ×3 |
| moe | model.11 | −0.0000 ×3 | 0.9375 ×3 |
| mot | model.14(.m.0/.m.1), model.20(.m.0/.m.1) | 0.6309 ×3 | 0.3333 ×3 |
| mot | model.23(.m.0/.m.1) | −0.0000 ×3 | 0.6667 ×3 |

三个 seed 下全部 15 个模块的指标**逐位一致**（4 位小数）。

## 解读（如实陈述）

1. **P0 smoke 度量的是未训练初始化状态的路由行为**（准入冒烟的本意是管线正确性，
   不是训练后路由形态）。该状态下指标由结构与确定性平局裁决主导，对 torch 随机
   seed 不敏感——这是观察到的性质，不是 bug；它同时说明单 seed 的历史 run 与
   本三 seed 证据兼容。
2. **MoE / 部分 MoT 模块的 Gini 恰为 (E−1)/E**（如 0.9375 = 15/16）：初始化路由
   完全坍缩到单专家（one-hot），熵为 0。这是未训练 argmax 路由的已知行为，
   也正是后续 P1 面板/训练实验要刻画的对象。
3. latent 模块 entropy_norm=1.0、gini=0：均匀分布锚点（与
   `tests/test_metric_analytic_groundtruth.py` 的解析解真值一致）。
