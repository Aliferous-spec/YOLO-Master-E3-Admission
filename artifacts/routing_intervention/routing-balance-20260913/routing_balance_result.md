# 路由平衡干预：正式实验结果（baseline coeff=1.0 vs moe_loss_fn.balance_loss_coeff=4.0）

- experiment id：`routing-balance-20260913`；**6/6 run 完成**（seeds 0/1/2 × 2 arms），无失败 run、无删除 run
- baseline checkout：`D:\YOLO-Master` @ `aa5d2e20c109b96f4a0c68f667ed2694586ef745`（未修改该目录，运行前后 git status 均空）
- 环境前提：`POLARS_SKIP_CPU_CHECK=1`；`PYTHONPATH=D:\YOLO-Master`
- 机器可读数据：`comparison.json`（统计与判据）、`epoch_metrics.csv`（180 行 = 6 run × 10 measured epoch × 3 layer）、`seed_summary.csv`、`experiment_environment.md`（环境 / 协议 / 时点）
- 本文件是生成器输出（`scripts/aggregate_routing_balance_experiment.py`）的正式化改写：**数字逐位未改**，只补上判据、饱和与限制的说明文字

## 判定

**NOT PASS，不支持在当前实验 regime 下 `balance_loss_coeff=4.0` 改善 routing balance。**

## 1. 预注册假设（preregistered hypothesis）

干预（唯一被允许改动的量）：三个目标 MoE layer（`model.5` / `model.8` / `model.11`）的
`moe_loss_fn.balance_loss_coeff`，baseline = 1.0 → intervention = 4.0。
不改 `module.balance_loss_coeff`（该属性不被辅助损失读取，改它只会误报「改了什么」），不改任何上游代码。

判据（实验前冻结，事后未调整）：

> intervention mean layer-Gini < baseline，且 paired bootstrap 95% CI 上界 < 0；
> 归一化熵方向一致（mean delta > 0 且 CI 下界 > 0）。

统计口径：以 **seed 为配对单位**（seed 0/1/2）。先算每个 arm 每个 seed 的 mean layer-Gini，
再对 3 个配对 delta 做 percentile bootstrap 95% CI（10000 resamples、固定 seed 0）。
不因结果调整 coeff / seed / epoch / 判据，也不挑选结果。

## 2. 观测结果（observed result）

逐 run 均值（10 measured epoch × 3 layer = 30 行 / run）：

| arm | seed | mean Gini | mean 归一化熵 | mean top1_share | wall s |
| --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.799038 | 0.185037 | 0.711618 | 28.76 |
| baseline | 1 | 0.854167 | 0.000000 | 1.000000 | 29.05 |
| baseline | 2 | 0.854167 | 0.000000 | 1.000000 | 46.19 |
| intervention | 0 | 0.800902 | 0.176769 | 0.726609 | 35.05 |
| intervention | 1 | 0.854167 | 0.000000 | 1.000000 | 74.60 |
| intervention | 2 | 0.854167 | 0.000000 | 1.000000 | 69.77 |

配对比较（intervention − baseline，seed 为配对单位）：

| metric | baseline 均值 | intervention 均值 | mean delta | paired bootstrap 95% CI | 逐 seed delta |
| --- | --- | --- | --- | --- | --- |
| layer-Gini | 0.835790 | 0.836412 | +0.000622 | [+0.000000, +0.001865] | seed0 +0.001865 / seed1 0.000000 / seed2 0.000000 |
| 归一化熵 | 0.061679 | 0.058923 | −0.002756 | [−0.008268, +0.000000] | seed0 −0.008268 / seed1 0.000000 / seed2 0.000000 |
| top1_share | 0.903873 | 0.908870 | +0.004997 | [+0.000000, +0.014991] | seed0 +0.014991 / seed1 0.000000 / seed2 0.000000 |

运行卫生：6 个 run 全部完成；系数断言共 84 条（每 run 14 条：`on_train_start` + 12 个
`on_train_epoch_start` + `on_train_end`）全部通过，baseline 恒为 1.0、intervention 恒为 4.0；
无 NaN / non-finite；无 callback 覆盖导致的系数漂移。

## 3. 统计判定（statistical decision）

| 判据条目 | 实测 | 是否满足 |
| --- | --- | --- |
| Gini mean delta < 0 | +0.000622 | 否 |
| Gini CI 上界 < 0 | +0.001865 | 否 |
| 熵 mean delta > 0 | −0.002756 | 否 |
| 熵 CI 下界 > 0 | −0.008268 | 否 |

**结论：NOT PASS，不支持在当前实验 regime 下 `balance_loss_coeff=4.0` 改善 routing balance。**
两个指标的方向都与假设相反（Gini 略升、归一化熵略降），且 CI 的上/下界都跨在 0 上。

## 4. 上限饱和解释（ceiling saturation）

Gini 的理论上限是 `(E−1)/E`（与 §4.1 / §4.5 的口径一致）：
`model.5` E=4 → 0.750000，`model.8` E=8 → 0.875000，`model.11` E=16 → 0.937500。

**180 个 measured row 中 143 行正好落在对应 layer 的 Gini 上限上**，同时归一化熵 0、top1_share 1.0
（即完全单专家坍缩）。逐 run 计数：

| run | 位于上限的行数 / 30 |
| --- | --- |
| baseline seed0 | 11 |
| baseline seed1 | 30 |
| baseline seed2 | 30 |
| intervention seed0 | 12 |
| intervention seed1 | 30 |
| intervention seed2 | 30 |

- seed 1 与 seed 2：两臂三层全部饱和，配对 delta 恰为 0 —— CI 里出现 `0.000000` 边界正是这个来源。
- seed 0 是唯一有 headroom 的 seed，全部 delta 都来自它：`model.8` baseline 0.760512 /
  intervention 0.760457（层贡献 −0.000018），`model.11` 0.886601 → 0.892250（层贡献 +0.001883），
  `model.5` 两臂都顶在 0.75（贡献 0），合计 +0.001865。即：**在唯一有可动空间的 seed 上，
  干预把 Gini 略微推高、把归一化熵略微压低。**
- 饱和不是「测量失败」：熵 0、top1_share 1.0、Gini 上限三者互相自洽，说明观测链在如实报告一个
  已经坍缩的路由状态。它的含义是——**本 regime 下该指标几乎没有可动空间**，因此本实验对
  「这个杠杆有没有用」的检验力很低。

## 5. 限制与不可过度解释（limitations）

本结果**不能**支持以下任何说法：

- 不声称该 intervention 在所有训练 regime 下无效 —— 本实验只有一个 regime（coco8 4 张图 /
  `yolo-master-n` @ 640 / CPU / batch 1 / 12 epochs，其中 2 warmup + 10 measured）。
- 不声称 balance loss 会导致普遍性能下降 —— 本实验不测 mAP、不测收敛、不测泛化。
- 不把 ceiling saturation 当成测量失败，也不因此宣布实验无效。
- 不把 null result 改写成成功；也不因结果不利而调整 coeff / seed / epoch / 判据或挑选结果。

其它限制：

- 结构化路由观测量 ≠ 精度结论；单 run wall-clock 28.76–74.60 s，measured epoch 计算合计 108.95 s。
- 有效配对样本只有 1 个（seed 1/2 完全饱和），因此 CI 的宽度与边界由饱和主导。
- 未做训练后 checkpoint 的复测，未覆盖 GPU / 分布式 / coco8 之外的数据集。

## 附：数据来源

- 逐 run：`seed{seed}_{arm}/intervention_result.json`（含 84 条系数断言、每 run 36 条 routing record、
  30 行 layer metric、起止时间与 wall-clock），原始记录 `seed{seed}_{arm}_records.jsonl`
- 汇总：`comparison.json` / `epoch_metrics.csv` / `seed_summary.csv`
- 环境与协议：`experiment_environment.md`
- 观测链：`scripts/routing_capture.py`（未修改，两臂完全相同）+ `_moe_force_snapshot`（两臂同开）
