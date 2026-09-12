# P1 训练 Slowdown：实验结果记录（seed 0/1/2）

判定依据：`docs/p1-judging-criteria.md`（训练 slowdown 预注册判据，冻结于正式实验之前）。

本文档**只记录并引用实际 artifact 数值**，不重新计算、不修改任何结果、不改判据。
所有数字均直接抄自
`artifacts/training_slowdown/train-slowdown-20260911/slowdown_result.json`。

## 1. workload / 环境 / baseline

| 项 | 值 |
| --- | --- |
| 模型 cfg | `ultralytics\cfg\models\master\v0_9\det\yolo-master-n.yaml`（相对 baseline checkout） |
| 数据 | `ultralytics\cfg\datasets\coco8.yaml`（相对 baseline checkout；coco8，4 张训练图 + 4 张验证图） |
| imgsz / batch / device / workers | 640 / 1 / cpu / 2 |
| Python | `C:\path\to\.venvs\yolo_master\Scripts\python.exe`（3.11.9） |
| `PYTHONPATH` | baseline checkout（YOLO-Master 上游仓库）（`import ultralytics` 解析到该 checkout 的 `ultralytics\__init__.py`） |
| torch / ultralytics | 2.13.0+cpu / 8.4.101 |
| platform | Windows-10-10.0.26200-SP0 |
| baseline_root | baseline checkout（YOLO-Master 上游仓库）@ `aa5d2e20c109b96f4a0c68f667ed2694586ef745` |
| official_base_ref | `3eb6cd914b651a06e2cd08ea87d12c28cab95502` |
| run_id | `train-slowdown-20260911` |
| 执行窗口 | 2026-09-10T23:30:45+08:00 → 2026-09-10T23:37:44+08:00 |

> **baseline 与 editable 是两个不同 checkout，不可视为同一个 commit。** 本次正式运行以
> `PYTHONPATH` 指向 baseline checkout 覆盖 venv 里的 editable 安装，使 `import ultralytics` 解析到
> **baseline** checkout（YOLO-Master 上游仓库 @ `aa5d2e20c109b96f4a0c68f667ed2694586ef745`），
> 而不是 P0 / P1-A 期间使用的 editable review checkout
> （另一个本地 checkout @ `d604c4bca8ceba3240c730f1b6e2767b7a320f6c`）。

ON 臂 = 训练 batch 结束后跑一遍 e3 观测链（MoE `_moe_force_snapshot=True` →
`routing_capture.capture_records` 发现与适配 → `RoutingRecordWriter` 追加 JSONL）；
OFF 臂 = 同一条训练循环，不跑观测链。两侧 workload 完全相同，唯一差异是观测链开关。

## 2. 设计：3 seeds / 26 epochs / 12 blocks / 6 pairs / ABBA

- **Seeds**：0 / 1 / 2（各一个真实训练 session，共 3 个）。
- **每个 seed 26 epochs** = 2 个未计量的 warmup epoch + 4 个 ABBA 块 × 6 epochs。
- **ABBA 块级配对**：块顺序 OFF → ON → ON → OFF，相邻配对（块 1&2、块 4&3）
  以抑制 seed 内的慢漂移。
- **12 blocks / 6 pairs**：3 seeds × 4 blocks = 12 个计量块；3 seeds × 2 pairs = 6 个配对观测。
- 每块 = 6 epochs × 24 个训练 batch（4 图 / batch=1）的 batch-loop 墙钟之和；
  逐 epoch 验证 / EMA / checkpoint 记账在计量窗口之外且两臂一致。
- ON 块每块记录 72 条 routing records，OFF 块 0 条；脚本对两侧都有硬校验
  （ON 块 0 记录或 OFF 块非 0 记录即中止），本次两侧均满足。

### 块级测时（秒，引自 `blocks[]`）

| seed | block | arm | epochs | batches | on_records | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | OFF | 6 | 24 | 0 | 19.20 |
| 0 | 2 | ON | 6 | 24 | 72 | 15.89 |
| 0 | 3 | ON | 6 | 24 | 72 | 13.98 |
| 0 | 4 | OFF | 6 | 24 | 0 | 18.47 |
| 1 | 1 | OFF | 6 | 24 | 0 | 20.21 |
| 1 | 2 | ON | 6 | 24 | 72 | 20.12 |
| 1 | 3 | ON | 6 | 24 | 72 | 18.03 |
| 1 | 4 | OFF | 6 | 24 | 0 | 19.83 |
| 2 | 1 | OFF | 6 | 24 | 0 | 16.32 |
| 2 | 2 | ON | 6 | 24 | 72 | 19.41 |
| 2 | 3 | ON | 6 | 24 | 72 | 19.23 |
| 2 | 4 | OFF | 6 | 24 | 0 | 19.25 |

### 配对 slowdown（引自 `pairs[]`）

| seed | pair | OFF block | ON block | slowdown_percent |
| --- | --- | --- | --- | --- |
| 0 | 1 | 1 (19.20s) | 2 (15.89s) | −17.25% |
| 0 | 2 | 4 (18.47s) | 3 (13.98s) | −24.30% |
| 1 | 1 | 1 (20.21s) | 2 (20.12s) | −0.45% |
| 1 | 2 | 4 (19.83s) | 3 (18.03s) | −9.07% |
| 2 | 1 | 1 (16.32s) | 2 (19.41s) | +18.91% |
| 2 | 2 | 4 (19.25s) | 3 (19.23s) | −0.10% |

## 3. 汇总统计（引自 `statistics.slowdown_percent`，n = 6）

| 统计量 | 值 |
| --- | --- |
| mean | −5.38% |
| median | −4.76% |
| std | 15.20 |
| min | −24.30% |
| max | +18.91% |
| p95 | +18.91% |
| bootstrap 95% CI | [−15.93%, **+6.12%**] |

bootstrap 口径（引自 `parameters.bootstrap`）：percentile bootstrap of the mean
（2.5%–97.5%），resamples = 10000，seed = 0。

## 4. 唯一预注册判据与判定

判据（引自冻结文档 `docs/p1-judging-criteria.md` §3）：**95% CI 上界 < 10%**，
且是唯一通过口径（不看点估计、不看单次 run、无人工判定）。

**判定：PASS** —— CI 上界 `+6.12%` < `10%`（`verdict.pass = true`，
`verdict.ci95_upper_percent = 6.118843913149156`）。

## 5. 必须如实披露的部分

- **`−5.38%` 不是「必然加速」。** 点估计为负只是本轮的均值方向；95% CI
  `[−15.93%, +6.12%]` **跨 0**，即本实验数据**不能排除「ON 与 OFF 无差异」**，
  更不能据此声称观测链会加速训练。本轮唯一成立的结论是判据 §4 的那一条：
  CI 上界 < 10%（未观察到 ≥10% 的训练减速）。
- 单观测散布（std 15.20）远大于效应量本身，配对值从 −24.30% 到 +18.91%
  跨越正负；其中 seed 2 的第一个配对是唯一正值（+18.91%）。这些失败方向 /
  异常波动一律保留在 artifact 中，未剔除、未重跑挑选。
- n = 6 个配对观测（3 seeds × 2 pairs），样本量小，CI 宽度直接反映这一点。

## 6. CPU 噪声限制（预先声明，见判据 §5）

- 环境为 CPU（`device=cpu`），计时受系统负载、频率、进程调度影响，
  slowdown 数值含测量噪声，bootstrap CI 反映的也是该噪声下的不确定性。
- 结论仅适用于本次 CPU 环境与上述 workload，不外推到 GPU / 分布式 /
  其他数据集或其他模型规模。

## 7. attempt1 的环境前置异常（发生在任何测量之前）

- 第一次启动（保留于 `artifacts/training_slowdown/attempt1-abort-polars/`）
  在 **epoch 0 之前**中止，未产生任何块级测量数据：
  基线 `trainer.read_results_csv()` → `import polars`，而 polars 的 CPU 探测在本机
  是**误报**（`platform.machine()` 返回空串 → 所有 feature flag 判为 unknown →
  `RuntimeError: unknown feature flag: 'sse3'`），并连带触发基线的
  "Initial training state is nonfinite; refusing to start without a healthy recovery checkpoint."
- **归类：环境前置问题（polars 与宿主 CPUID 探测不匹配），不是实验测量失败，
  也不是被测观测链的开销结果。**
- 处置：仅设置 polars 官方开关 `POLARS_SKIP_CPU_CHECK=1`（绕过误报），
  随后一次性完整重跑。**协议、判据、workload、seed、配对方式、统计方法、
  被测链路与基线代码均未改动**，也未按结果重跑挑选。
- attempt1 的残留（`run.log` / `run.err.log` / 空 `seed0_on_records.jsonl` /
  `runs\seed0\args.yaml`）**原样保留**，未被覆盖或删除。

## 8. artifact 路径

运行目录：`artifacts/training_slowdown/train-slowdown-20260911/`

| 文件 | 内容 |
| --- | --- |
| `slowdown_result.json` | 最终结果：protocol / parameters / baselines / environment / blocks / pairs / statistics / verdict |
| `seed0_on_records.jsonl` | seed 0 原始 ON 记录（144 行 = 2 个 ON 块 × 72 条） |
| `seed1_on_records.jsonl` | seed 1 原始 ON 记录（144 行） |
| `seed2_on_records.jsonl` | seed 2 原始 ON 记录（144 行） |
| `run.log` | 完整运行日志（含每块测时与最终统计输出） |
| `run.err.log` | 空文件（本次无 stderr 输出） |
| `runs/seed{0,1,2}/args.yaml` | 各 seed 的完整训练参数 |
| `runs/seed{0,1,2}/results.csv` | 各 seed 的逐 epoch 训练指标 |
| `runs/seed{0,1,2}/weights/{best,last,last_healthy}.pt` | 训练产物权重；**不纳入 Git**，仅为本地训练产物（见下注） |

中止的第 1 次尝试：`artifacts/training_slowdown/attempt1-abort-polars/`（证据保留）。

> 注：`runs/seed*/weights/*.pt` 是本地训练产物，**明确不纳入 Git**
> （由 `.gitignore` 的 `artifacts/**/weights/` 排除），仅保留在本机工作目录，
> 不随仓库分发、不参与证据清单校验。除此之外的其余证据
> （`slowdown_result.json`、`seed*_on_records.jsonl`、`run.log`、`run.err.log`、
> `runs/seed*/args.yaml`、`runs/seed*/results.csv` 等）仍按上述 artifact 清单纳入保存。

## 9. 复现命令

工作目录：本仓库根 `C:\path\to\YOLO-Master-E3-Admission`（脚本会因输出已存在而拒绝覆盖；重跑请加 `--overwrite`
或换 `--run-id`）：

```powershell
$env:PYTHONPATH = "C:\path\to\YOLO-Master"
$env:PYTHONUTF8 = "1"
# polars 官方开关，绕过本机 CPUID 误报（见 §7；不影响协议与测量）
$env:POLARS_SKIP_CPU_CHECK = "1"
& "C:\path\to\.venvs\yolo_master\Scripts\python.exe" scripts/measure_training_slowdown.py `
    --baseline-root C:\path\to\YOLO-Master `
    --run-id train-slowdown-20260911
```

只做配置校验、不训练：

```powershell
$env:PYTHONPATH = "C:\path\to\YOLO-Master"
& "C:\path\to\.venvs\yolo_master\Scripts\python.exe" scripts/measure_training_slowdown.py `
    --baseline-root C:\path\to\YOLO-Master --dry-run
```
