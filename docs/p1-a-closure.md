# E3 P1-A 逐样本采集 Closure（P1-A Evidence Closure）

Closure 时间：2026-09-07 00:23–00:24（UTC+8，Asia/Shanghai）；文档定稿于 2026-09-07。
Closure 对象：`C:\tmp\e3-package`（`Aliferous-spec/YOLO-Master-E3-Admission`，P1-A 实施为 HEAD `47c0c44`（P0 closure）之上的未提交改动，本次 commit 一并封存）。
验收命令基线：`C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe`（Python 3.11.9 / torch 2.13.0+cpu / ultralytics 8.4.101）。

> 结论先读：P1-A 全部验收项均为 **PASS**，无 FAIL / BLOCKED。P1-B（逐样本开销测量）**尚未开始**。所有结论均来自本机实际执行与产物复核，无猜测项。

---

## 1. run_id 与产物

- 正式 P1-A smoke run_id：`smoke-20260907-002319-d05c10`。
- 产物目录：`artifacts/smoke/smoke-20260907-002319-d05c10/`（14 个文件：P0 型 13 文件 + P1 新增 `sample_routing_records.jsonl`）。
- smoke 命令（在 `C:\tmp\e3-package`，exit 0，`result=PASS`）：

```
C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe scripts\run_e3_smoke.py --config configs\e3_smoke.yaml --baseline-root D:\YOLO-Master
```

- 运行区间：`full.log` 2026-09-07 00:23:19 → 00:23:49；`environment.json`：Windows 10 / Python 3.11.9 / torch 2.13.0+cpu / ultralytics 8.4.101；`official_base_ref: 3eb6cd914b651a06e2cd08ea87d12c28cab95502`。

## 2. P0 regression 核对 — 无回归

- P0 四 step（mot / moe / latent / overhead）在本次正式 run 中全部 **PASS**（`summary.json` 各 step `status=PASS`）。
- P0 v1 capture 语义未变：canonical `routing_records.jsonl` 15 行（MoT 9 + MoE 3 + Latent 3），每 family 每模块一行，全部 `step=None`（无任何 `step` 为 int 的行），与 P0 acceptance 语义一致。
- P0 evidence / P0 acceptance（`docs/p0-acceptance.md`）/ schema（`routing_record.py`）/ adapter / writer / 上游 YOLO-Master 均未修改；实施改动仅新增代码路径（`git diff --stat` 相对 HEAD：`routing_capture.py` +92、`run_e3_smoke.py` +261，0 删除行）。
- 全量 pytest（68 passed，含 P0 既有 56 项）证明 P0 行为无回归，见 §6。

## 3. sample rows 统计

`sample_routing_records.jsonl`（75,051 bytes，51 行）为 sample rows 的唯一落盘位置：

| family | samples（step 0..S-1） | modules/step | rows（S x M） |
| --- | --- | --- | --- |
| mot | 4 | 9 | 36 |
| moe | 4 | 3 | 12 |
| latent | 1 | 3 | 3 |
| 合计 | — | — | 51 |

- P1-A sample step 日志：`mot: samples=4 modules=9 records=36`、`moe: samples=4 modules=3 records=12`、`latent: samples=1 modules=3 records=3`，`P1-A sample rows written: 51`。
- data-flow 约束（spec §3.1）：sample step 未向 `execute_smoke_steps()` 返回 `records`（`summary.json` 中 samples step 无 records 计数），canonical 文件不含任何 sample row。

## 4. schema / step / run_id 校验

独立审计（全量逐行 JSON 复核，语义同 `verify_sample_records` / `iter_records`）结果：

- schema：sample 与 canonical 全部行 `schema_version == "e3-routing/v1"`，违规计数 0。
- run_id：sample 51 行与 canonical 15 行的 `run_id` 全部等于 `smoke-20260907-002319-d05c10`，与目录名一致。
- step：所有 sample 行 `step` 为 int>=0；per family `step` 连续（mot/moe: 0..3；latent: 0）；同 family 每 step 模块集一致；`(run_id, family, module.name, step)` 主键无重复。
- 行数：mot 36 = 4x9、moe 12 = 4x3、latent 3 = 1x3，合计 51，与 S x M 期望一致。
- canonical 纯净性：canonical 15 行均无 int `step`，且没有任何 canonical 行与 sample 主键重叠（sample rows 未混入 canonical）。

## 5. manifest / verify 结果

- `manifest.sha256.json`：13 项（覆盖目录内除自身外全部 13 个文件）。
- 独立 SHA-256 复核：13/13 一致，无缺失、无篡改。
- `--verify-artifacts`（exit 0）：`result=PASS manifest OK`。命令：

```
C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe scripts\run_e3_smoke.py --verify-artifacts artifacts\smoke\smoke-20260907-002319-d05c10
```

- `summary.json`：`status=PASS`，`validation.errors=[]`；`full.log` 以 `result=PASS` 结束。

## 6. pytest / ruff 结果

- 全量 pytest（在 `C:\tmp\e3-package`）：`68 passed in 1.47s`，exit 0。覆盖既有 P0 套件（56 项）与 P1 新增 `tests/test_p1_sample_capture.py`（12 项）。
- ruff（venv 内 `ruff 0.16.4`），本次涉及文件：`scripts/routing_capture.py`、`scripts/run_e3_smoke.py`、`tests/test_p1_sample_capture.py` → `All checks passed!`，exit 0。

## 7. P1-B 状态 — NOT STARTED（明确）

- P1-B（逐样本采集 overhead 测量，spec §7）**尚未开始**：本 closure 未做任何 overhead 实验，产物目录不含 `sample_overhead_result.json`，未引用或混入 P0 `overhead_result.json` 数值。
- P0 overhead step 的 `overhead_percent=-0.39` 仅为 P0 型证据（hook 开关计时波动，P0 只校验有限值），不代表 P1-B 结果。
- 下一步（未在本 commit 范围内）：P1-B 测量协议、预注册阈值/置信区间方法、独立 `sample_overhead_result.json` 证据。

---

## 8. 边界说明（如实记录，不构成 FAIL/BLOCKED）

- P1-A 验收项依据 `docs/p1-spec.md` §3/§8/§9（P1-A 能力项 1–5）；P1-B 验收项（§9 8–10）未执行。
- 运行时 `ultralytics` 包解析自 venv editable 安装；smoke `baseline_root` 为 `D:\YOLO-Master`；`official_base_ref` 为配置记录值，环境实况同 P0 acceptance §4，本 closure 不重述结论。
- 本 closure 不冻结跨族统一 schema，不提交上游 PR，不扩展样本集/数据集/checkpoint。