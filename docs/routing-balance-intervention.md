# 路由平衡干预：`moe_loss_fn.balance_loss_coeff` 1.0 vs 4.0（正式实验，NOT PASS）

- 实验 id：`routing-balance-20260913`。
- 唯一数值真值源：`artifacts/routing_intervention/routing-balance-20260913/comparison.json`；
  本文只转述该 artifact，不重算、不改任何数字。
- 判定：**NOT PASS**（预注册判据未满足，如实记录为 null result）。

## 1. 动机

P0/P1 的只读观测只能回答「现在路由长什么样」，不能回答「动一个旋钮能不能改变它」。
自然的下一个问题是：**提高 MoE 平衡损失的权重，能不能降低路由失衡？**
这是与温度探针不同的杠杆（那里调温度 API，这里调 `moe_loss_fn.balance_loss_coeff`），
沿用同一条纪律：先冻结判据、后执行实验。

## 2. 干预（intervention）

- 只改三个目标 MoE layer（`model.5` / `model.8` / `model.11`）的
  `moe_loss_fn.balance_loss_coeff`：**baseline = 1.0 → intervention = 4.0**。
- 不改 `module.balance_loss_coeff`（该属性不被辅助损失读取，改它只会误报「改了什么」）；
  不修改 YOLO-Master 上游仓库任何代码。
- 系数断言：`on_train_start` / 每个 `on_train_epoch_start` / `on_train_end` 各断言一次，
  84/84 全绿（baseline 恒 1.0、intervention 恒 4.0）；无 NaN / non-finite，无 callback 覆盖。

## 3. 协议（protocol）

- **3 seeds × 2 arms**（seed 0/1/2 × baseline/intervention）= 6 个 run，6/6 完成，无失败、无删除。
- 数据 coco8：**4 张训练图 + 4 张验证图**。
- **640 / batch 1 / CPU / workers 0**。
- **12 epochs = 2 warmup + 10 measured**；两臂使用同一条未修改的 `scripts.routing_capture`
  观测链，并同时开启 `_moe_force_snapshot`。
- 逐 epoch 逐 layer 记录：**Gini**（`load_gini`）、**归一化熵**（`routing_entropy_normalized`）、
  **top1 share**（`dominant_expert_share`）、dominant expert、module aux loss、
  模型 `_last_mixture_aux_loss`、epoch wall-clock。
- 环境：Python 3.11.9 / torch 2.13.0+cpu / ultralytics 8.4.101 / Windows（纯 CPU）；
  运行前提 `POLARS_SKIP_CPU_CHECK=1`；baseline checkout 运行前后 `git status` 均空。

## 4. 预注册判据（事后未调整）

以 **seed 为配对单位**（0/1/2），统计口径为 **paired bootstrap 95% CI**
（percentile bootstrap，10000 resamples，固定 seed 0）：

> intervention mean layer-Gini < baseline，且 CI 上界 < 0；
> **归一化熵**方向一致（mean delta > 0 且 CI 下界 > 0）。

## 5. 结果（observed result）

逐 run 均值（10 measured epoch × 3 layer = 30 行 / run）：

| arm | seed | mean Gini | mean 归一化熵 | mean top1_share |
| --- | --- | --- | --- | --- |
| baseline | 0 | 0.799038 | 0.185037 | 0.711618 |
| baseline | 1 | 0.854167 | 0.000000 | 1.000000 |
| baseline | 2 | 0.854167 | 0.000000 | 1.000000 |
| intervention | 0 | 0.800902 | 0.176769 | 0.726609 |
| intervention | 1 | 0.854167 | 0.000000 | 1.000000 |
| intervention | 2 | 0.854167 | 0.000000 | 1.000000 |

配对比较（intervention − baseline，seed 为配对单位）：

| metric | baseline 均值 | intervention 均值 | mean delta | paired bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| layer-Gini | 0.835790 | 0.836412 | +0.000622 | [+0.000000, +0.001865] |
| 归一化熵 | 0.061679 | 0.058923 | −0.002756 | [−0.008268, +0.000000] |
| top1_share | 0.903873 | 0.908870 | +0.004997 | [+0.000000, +0.014991] |

判据两条都不满足（Gini delta > 0 且 CI 上界 +0.001865 > 0；熵 delta < 0）→
**NOT PASS：不支持在当前实验 regime 下 `balance_loss_coeff=4.0` 改善 routing balance。**
Gini 略升、归一化熵略降，CI 的上/下界都跨在 0 上。

## 6. 天花板饱和（ceiling saturation，为什么检验力低）

Gini 的理论上限是 `(E−1)/E`：`model.5` E=4 → 0.750000、`model.8` E=8 → 0.875000、
`model.11` E=16 → 0.937500。

**180 个 measured row 中 143 行正好落在对应 layer 的 Gini 上限上**（同时归一化熵 0、
top1_share 1.0，即完全单专家坍缩）。逐 run 落在上限的行数：baseline seed0 = 11、
intervention seed0 = 12，**seed 1 / 2 两臂各 30（全部饱和）**。

- **seed 1 / 2 两臂三层全部饱和**，配对 delta 恰为 0 —— CI 里出现的 `0.000000` 边界即来自这里。
- 全部 delta 来自 seed 0（唯一有 headroom 的 seed）：`model.11` 由 0.886601 升到 0.892250，
  `model.8` 基本不变（0.760512 → 0.760457），`model.5` 两臂都顶在 0.750000，合计 +0.001865。
- 饱和不是「测量失败」：熵 0、top1_share 1.0、Gini 上限三者互相自洽，说明观测链如实报告了
  一个已经坍缩的路由状态。它的含义是——**本 regime 下该指标几乎没有可动空间**，
  因此本实验对这个杠杆的检验力很低。

## 7. 当前 regime 的限制（limitations）

- 本结果只覆盖**单一 regime**：coco8 4 张训练图 + 4 张验证图 / `yolo-master-n` @ 640 /
  CPU / batch 1 / workers 0 / 12 epochs（2 warmup + 10 measured）。
- 不测 mAP、不测收敛、不测泛化；未对训练后 checkpoint 复测。
- 未覆盖 GPU / 分布式 / coco8 之外的数据集。
- 有效配对样本实际上只有 1 个（seed 1/2 完全饱和），CI 的宽度与边界由饱和主导。
- 因此**不能**声称该干预在所有训练 regime 下无效，也**不能**声称 balance loss 会造成
  普遍性能下降；只能如实报告当前 regime 下的 null result，不调参重跑、不改写为成功。

## 8. 证据入口

`artifacts/routing_intervention/routing-balance-20260913/`：

- `comparison.json` —— 统计与判据（唯一数值真值源）
- `epoch_metrics.csv`（180 行 = 6 run × 10 measured epoch × 3 layer）、`seed_summary.csv`
- `experiment_environment.md` —— 环境 / 协议 / 时点
- `routing_balance_result.md` —— 正式化结果说明
- 逐 run：`seed{seed}_{arm}/intervention_result.json` 与 `seed{seed}_{arm}_records.jsonl`
