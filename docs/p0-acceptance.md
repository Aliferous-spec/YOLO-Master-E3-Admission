# E3 P0 最终验收记录（P0-6 Acceptance）

验收时间：2026-09-05 20:45–20:47（UTC+8，Asia/Shanghai）
验收对象：`C:\tmp\e3-package`（`Aliferous-spec/YOLO-Master-E3-Admission`，工作树包含 P0-1..P0-5 的未提交改动）
验收命令基线：`C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe`（Python 3.11.9 / torch 2.13.0+cpu / ultralytics 8.4.101）

> 结论先读：以下全部验收项均为 **PASS**，无 FAIL / BLOCKED。所有结论均来自本机实际执行与产物复核，无猜测项。

---

## 1. P0 逐项状态

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| P0-1 三族 collector 接通 | **PASS** | `scripts/routing_capture.py` 复用现有 adapter/writer/schema；真实 smoke 中 MoT/MoE/Latent 每族 forward 后各生成 v1 记录并统一写入 `routing_records.jsonl`（见 §3）。单测：`tests/test_routing_capture.py`（发现/适配/单文件 JSONL/合并 JSONL）。 |
| P0-2 失败隔离 | **PASS** | `execute_smoke_steps()` 逐族独立执行，异常只标记该 step 为 FAIL，后续族照常运行；summary 含 `started_at/finished_at/status/error` 及每 step 时间戳。单测：`tests/test_run_e3_smoke.py::test_execute_smoke_steps_does_not_block_later_steps`、`..._records_times`。 |
| P0-3 run_id 与产物隔离 | **PASS** | `--run-id` 优先；缺省时 `generate_run_id()` 自动生成唯一 id。本次真实 smoke 即使用自动 id `smoke-20260905-204546-6c7389`。每次运行落在独立 `artifacts/smoke/<run_id>/`，manifest 只含本 run 文件。单测覆盖重复运行隔离。 |
| P0-4 overhead 与 logging | **PASS** | `parse_overhead_percent()` 支持 `-5.01%`/`+1.20%`/`0.99%` 等带符号值，拒绝 `nan/inf`；本次实测 `-9.44%`（首跑）与 `21.75%`（验收跑）均被正确解析为有限值。`setup_logging()` 重复调用不产生重复 handler。单测：`test_parse_overhead_percent_*`、`test_setup_logging_*`。 |
| P0-5 manifest verify | **PASS** | 独立 `verify_manifest()` + CLI `--verify-artifacts`。验收跑 verify 结果为 `result=PASS manifest OK`（exit 0），独立 SHA-256 复核 12/12 一致。单测覆盖 manifest 缺失 / 文件缺失 / 文件被篡改 / run_id 不一致。 |
| P0-6 最终验收 | **PASS** | 见 §2 全部八项检查；本文档即交付物。 |

---

## 2. P0-6 验收检查清单（严格标记）

### 2.1 全部 pytest — **PASS**
命令（在 `C:\tmp\e3-package`）：
```
C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe -m pytest tests -q
```
结果：`56 passed in 1.55s`，exit 0。覆盖文件：`test_routing_capture.py`、`test_run_e3_smoke.py`、`test_routing_record.py`、`test_routing_record_writer.py`、`test_moe_adapter.py`、`test_mot_adapter.py`、`test_latent_adapter.py`、`test_smoke_contract.py`。

### 2.2 一次最小三族真实 smoke — **PASS**
命令（自动生成 run_id，验证 P0-3 默认路径）：
```
C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe scripts\run_e3_smoke.py --config configs\e3_smoke.yaml --baseline-root D:\YOLO-Master
```
结果：exit 0，日志与终端均以 `result=PASS` 结束。run_id：`smoke-20260905-204546-6c7389`。产物目录：`artifacts\smoke\smoke-20260905-204546-6c7389\`。

### 2.3 manifest verify — **PASS**
命令：
```
C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe scripts\run_e3_smoke.py --verify-artifacts artifacts\smoke\smoke-20260905-204546-6c7389
```
结果：exit 0，`result=PASS manifest OK`。独立复核（另行计算 SHA-256）：manifest 12 项，全部存在且哈希一致，无缺失、无篡改。

### 2.4 三族自动发现 — **PASS**
`routing_records.jsonl` 中 15 条记录按族分布（由 `discover_routed_modules()` 在真实 forward 后从 `last_routing_snapshot` 自动发现）：

- MoT 9 个模块：`model.14 / model.14.m.0 / model.14.m.1 / model.20 / model.20.m.0 / model.20.m.1 / model.23 / model.23.m.0 / model.23.m.1`
- MoE 3 个模块：`model.5 / model.8 / model.11`（routing 层）
- Latent 3 个模块：`model.23 / model.24 / model.25`

### 2.5 三族 v1 JSONL — **PASS**
`routing_records.jsonl` 共 15 行，逐行解析：`schema_version == "e3-routing/v1"` 且 `run_id == smoke-20260905-204546-6c7389`，违规计数 0。MoT 9 + MoE 3 + Latent 3 = 15，覆盖三族。

### 2.6 Latent 真实 snapshot — **PASS**
`latent_snapshot.jsonl` 3 条真实记录（`model.23/24/25`），每条 `num_keys = 36`；快照键为真实字段（抽样：`active_experts_per_sample`、`aux_loss`、`balance_loss`、`dispatch_policy`、`entropy` 等）。对应 v1 记录亦含 36 个 `source_snapshot_keys`，来自同一真实 forward。

### 2.7 失败隔离 — **PASS**
- 单测证明：MoE step 抛错后 Latent/overhead 仍执行（`test_execute_smoke_steps_does_not_block_later_steps`）；PASS/FAIL step 均记录 `started_at/finished_at/status`，失败 step 与顶层 summary 记录 `error`。
- 真实 run 的 `summary.json` 结构验证：顶层 `started_at=2026-09-05T20:45:47+08:00`、`finished_at=2026-09-05T20:46:40+08:00`、`status=PASS`；四 step（mot/moe/latent/overhead）均为 PASS 且各带起止时间。

### 2.8 产物完整性 — **PASS**
`artifacts\smoke\smoke-20260905-204546-6c7389\` 共 13 个文件：`config.resolved.yaml`、`environment.json`、`full.log`、`latent_snapshot.jsonl`、`manifest.sha256.json`、`moe_usage_stats.json`、`mot_deformable_activation_check.csv`、`mot_expert_heatmap_top1_share.png`、`mot_routing_detailed.csv`、`mot_routing_scenarios.csv`、`overhead_result.json`、`routing_records.jsonl`、`summary.json`。manifest 覆盖其中 12 项（自排除 manifest 自身），SHA-256 全数一致。

---

## 3. 执行中发现并修复的问题

**Manifest 与 full.log 的哈希时序 bug（P0-5 验收时发现，已修复）**

- 现象：对首个新鲜 run（`smoke-20260905-203946-f1560a`）执行 `--verify-artifacts` 返回 `sha256 mismatch: ...\full.log`，独立复核一致地仅 full.log 失配。
- 根因：`main()` 先写 manifest，之后才 `logger.info("result=PASS")`，导致 full.log 在哈希后又追加一行。
- 修复（`scripts/run_e3_smoke.py`）：把最终 `result=` 日志移到写 manifest 之前，并在写 manifest 前 flush + close 文件 handler，使 manifest 哈希到 full.log 的最终状态。
- 验证：修复后重新执行全部 pytest（56 passed）与一次真实 smoke（`smoke-20260905-204546-6c7389`），`--verify-artifacts` 返回 PASS，独立 SHA-256 复核 12/12 一致。
- 说明：修复前生成的旧产物（`admission-20260825`、`p0-acceptance-20260905`、`smoke-20260905-203946-f1560a`）受该时序影响，其 full.log 哈希不保证可复核；验收证据以修复后的 `smoke-20260905-204546-6c7389` 为准。

---

## 4. 环境与边界说明（如实记录，不构成 FAIL/BLOCKED）

- overhead 实测值：首跑 `-9.44%`、验收跑 `21.75%`（hook 开关时间差波动）。P0-4 只要求解析带符号值并拒绝 NaN/Inf，P0 验收清单未包含 `<10%` 阈值判定，故不据此判 FAIL。
- 运行时 `ultralytics` 包解析自 venv 的 editable 安装 `D:\Claude_Workspace\projects\YOLO-Master-review`（HEAD `d604c4b`，工作树含未提交改动）；smoke 的 `baseline_root`（chdir 目标、harness 脚本与 model config 来源）为 `D:\YOLO-Master`（HEAD `aa5d2e2`）；`configs/e3_smoke.yaml` 内 `official_base_ref: 3eb6cd9...` 为配置记录值，与上述两个 HEAD 不一致，属环境实况，仅记录备查。
- 验收结论不涉及：统一 schema 正式冻结、训练减速正式结论、上游 PR。