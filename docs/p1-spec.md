# E3 P1 Spec（最小可执行版）

- 状态：**P1-A 已实现并通过 closure（2026-09-07）**，见 `docs/p1-a-closure.md`；P1-B 仍未实现、尚未开始。
- 依据：`C:\tmp\e3-package` @ HEAD `47c0c44`（P0 已 closure）。P0 acceptance、P0 evidence、`RoutingRecord e3-routing/v1`、历史 overhead 数字全部冻结，本 spec 不触碰。
- 本文件为 P1（A/B）定义基线；P1-A 实施与验收见 `docs/p1-a-closure.md`，P1-B 仍为 spec-only。
- 规则：凡本 spec 未定义或与冻结对象冲突的需求，一律停下请示，不得在实现中擅自扩展。

## 1. P1 目标

把 E3 路由证据的粒度从“每次 run 的模块级快照”（P0）扩展到“每个 sample（每个输入）的模块级快照”（P1），并配套可复现的逐样本采集测量与入库证据。P1 不改变 P0 四 step 的行为与产物，不冻结跨族统一 schema，不修改上游 YOLO-Master，不提交上游 PR。

保守范围（不引入新数据/新任务）：MoT 4 类合成场景各 1 图、MoE coco8 val 4 张图（batch=1）、Latent 1 张 640×640 随机图。
理由：只验证“逐样本采集 + 测量 + 证据”闭环在既有输入集合上成立；样本集扩展、真实 checkpoint、MOT 评估均不在 P1。

输入澄清（P0 v1 capture 与 P1 sample 的关系，重要）：

- P0 生成 `routing_records.jsonl` 的 v1 capture 实际输入是**每 family 单次前向**：MoT/Latent = 1 张随机 640×640 图（eval，`training=False`）；MoE = 1 张随机 640×640 图（train-mode，`training=True`）。P0 v1 capture **没有**逐样本/逐场景前向。
- P0 的 CSV diagnostic（`scripts/diagnose_mot_routing.py --synthetic` 的 4 类合成场景）与 MoE `ExpertUsageTracker` 的 coco8 val 只进入 CSV / `moe_usage_stats.json` 等非 v1 证据，**不是** P0 v1 capture 的输入。
- P1 sample 集合取自上述“非 v1 证据的既有输入集合”（MoT 合成场景、MoE coco8 val），加上 Latent 的既有单张随机图。因此 P1 逐样本前向是**新增路径**，不能把 P0 v1 capture 的代码路径理解为“每样本执行一次”即可（P0 v1 capture 只有单次随机前向）。

## 2. 术语

- sample：一次独立前向输入（MoT=1 张合成场景图；MoE=1 张 coco8 val 图；Latent=1 张 640 随机图）。
- module-level row：一条 `e3-routing/v1` `RoutingRecord`，粒度 = module × 一次 forward。
- P0 v1 capture：P0 中生成 `routing_records.jsonl` 的采集路径（每 family 单次前向，输入见 §1）。
- aggregate rows（P0 语义）：`routing_records.jsonl` 中 `step=None`、每模块一行（P0 v1 capture 产生）。
- sample rows（P1 语义）：写入 `sample_routing_records.jsonl`、`step=k`、每模块一行（P1-A 逐样本 capture 产生）。
- P0 非 v1 证据输入：MoT CSV diagnostic 的 4 类合成场景、MoE `ExpertUsageTracker` 的 coco8 val；只进入 CSV/`moe_usage_stats.json`，不进入 P0 v1 capture。

## 3. P1-A 范围（逐样本采集能力）

### 3.1 数据流（强制约束）

- sample rows **只写** `sample_routing_records.jsonl`（独立路径）。
- sample rows **不得**通过 P1-A step 的返回 `records` 暴露：`execute_smoke_steps()` 会把每个 step 返回的 `records` 汇总进 `collected`，并在 `main()` 末尾统一写入 `routing_records.jsonl`。因此 P1-A step 必须在其内部直接完成 `sample_routing_records.jsonl` 的写盘（复用 `RoutingRecordWriter` / `write_records` 指向该独立路径），并返回空 `records`。
- `routing_records.jsonl` 只保留 P0 四 step 产生的 `step=None` 行，语义与 P0 完全一致。

### 3.2 每 family 的 sample 源与逐样本前向

1. 每个 family 按固定顺序逐个 sample 前向；每个 sample 后对已发现模块生成 v1 记录（复用 `routing_capture` 的 discovery + 现有三个 adapter + `RoutingRecordWriter`），`step` 设为该 family 内 0-based sample 序号。
2. 各 family 样本源与模式（均为 P1 新增的逐样本循环，不是 P0 v1 capture 的“单次随机前向”）：
   - MoT：样本源 = 上游 `synthetic_scenes` 生成的 4 张 640×640 合成场景图（顺序 sparse/dense/large_regular/irregular_occluded；与 P0 CSV diagnostic 同一输入集合）；逐张 **eval** 前向（`training=False`，与 P0 v1 MoT capture 的模式开关一致）；每张后 `capture_records(step=i)`。
   - MoE：样本源 = coco8 val 的 4 张图（batch=1；与 `ExpertUsageTracker` 的 usage 证据同一输入集合），预处理方式与 YOLO val 一致（letterbox 到 640 后取单张张量）；逐张 **train-mode** 前向（上游 MoE `last_routing_snapshot` 仅在 `self.training` 分支发布，见 `ultralytics/nn/modules/moe/gated.py` / `modules.py` 的 `_record_moe_snapshot` 调用点），并设置 `_moe_force_snapshot=True`（或等价使每个 sample 都刷新快照）；每张后 `capture_records(training=True, step=i)`。
   - Latent：样本源 = 1 张 640×640 随机图（与 P0 v1 Latent capture 相同输入）；eval 前向，`step=0`。
3. P0 四 step（mot/moe/latent/overhead）及其产物保持原样、照常运行。

### 3.3 MoE 连续 train-mode 的状态隔离（BN 为主）

连续 train-mode 前向会让 BatchNorm 的 running statistics 随 sample 顺序累积漂移，进而影响后续 sample 的路由输入与结果。定义如下：

- 首选方案（每个 sample 前恢复 BN running state）：首个 MoE sample 前，对模型内所有 `track_running_stats=True` 的 BatchNorm 模块记录初始 `running_mean` / `running_var` / `num_batches_tracked` 快照；每个 sample 前向之前把该快照原位写回，再做该 sample 的 train-mode 前向。实现只在 e3 侧（不改上游），开销为 O(BN buffers) 的拷贝，可忽略；效果是每个 sample 从同一 BN 状态出发，路由结果与 sample 顺序无关。yolo-master-n 的 Conv 默认 `bn=True`（含标准 BatchNorm），该方案可安全实现。
- 兜底替代（仅当首选方案在具体模型上无法安全执行时）：放弃恢复，按固定顺序逐 sample 前向，用 `step` 序号标识行。代价：BN running 状态随 sample 顺序漂移，路由值顺序相关，不能宣称 order-invariant；P1-A 文档必须显式声明采用了该兜底。此替代仅作异常兜底，不作为默认路径。
- 已知边界：train-mode 下 MoE 的内部副作用（如逐模块 `training_step` 计数、任何未处于关闭态的 noise/dropout/balance 逻辑）不在上述 BN 恢复范围内。P1-A 单测必须断言该模型上这些副作用为关闭/惰性状态；若发现它们影响路由值，则把相应状态纳入恢复集合，或转用兜底替代并记录。

实现期允许修改：`scripts/run_e3_smoke.py`（新增 sample step 或增量 helper）、`scripts/routing_capture.py`（新增 per-sample 循环 helper，纯增量）、`tests/`、文档。
实现期禁止修改：`scripts/routing_record.py`、三个 `*_adapter.py`、`RoutingRecordWriter` 语义、`configs/e3_smoke.yaml`（本 spec 不新增 config 键；如需新增须另行决策）、P0 产物/验收/历史数字、上游代码。

## 4. P1-B 范围（测量 + 验收证据）

1. 新增逐样本测量脚本（对标 `measure_routing_hook_overhead.py` 的结构，但被测对象是 P1-A 链路）：OFF=纯 forward；ON=逐样本采集路径（与 P1-A 实现一致，MoE 含 BN 状态恢复与 snapshot force 逻辑）。
2. 以独立 step 加入 P1 run，失败不影响其他族（沿用隔离）。
3. 输出 `sample_overhead_result.json`，含协议元数据与统计量。
4. 执行一次完整 P1 smoke（P0 四 step + P1-A step + P1-B step），生成 run 产物并通过 verify。
5. 验收记录按 P0 模式单独成文（如 `docs/p1-acceptance.md`），属于 P1 交付范围。

## 5. sample identity 决策（保守方案）

- 身份主键：`(run_id, family, module.name, step)`。
- `step` = 该 family 内 0-based sample 序号（0..S_f-1），复用 `e3-routing/v1` 已有的可选整数字段 `step`（schema 校验为 int≥0，`capture_records` 已支持透传）。
- P0 行 `step=None` 语义保留为“run 级单次前向的模块发现行”；sample rows 一律 `step=int`。该定义只澄清既有可选字段语义，不改 schema 文件。
- 不在 v1 记录内新增 image_id/scene/batch_index 等字段（不改 adapter 白名单、不改 schema）。需要外部 id（如 MoT CSV 的 image_id）时，用“同 run 同 family 的相同顺序”按 ordinal 关联，关联关系不写入 v1，作为文档化限制。
- 文件决策：sample rows 写入**独立新文件** `sample_routing_records.jsonl`，不混入 `routing_records.jsonl`（数据流约束见 §3.1）。
  理由：保持 canonical 文件既有语义（module-level 单次发现）不被新粒度污染；P0 侧阅读/校验零影响；writer 与 schema 完全复用，唯一新增是一个文件路径。
- 保守性说明：以上是当前冻结约束下改动面最小的方案；若后续需要结构化 per-sample 身份（image/scene），必须单独决策（扩展 adapter 白名单或 bump schema），不属于 P1。

## 6. schema compatibility 原则

- `e3-routing/v1` 冻结：不新增/删除/改名任何 key，不 bump 版本，`RoutingRecord` 构造与反序列化校验逻辑不变。
- P1 sample rows 仍是合法 v1 行：`iter_records` 可无错解析，字段齐全，`run_id` 与 run 一致。
- `step` 语义由本 spec 定义；P0 中该字段无正式语义，本定义不修改 schema 文件。
- 数据流分离是兼容原则的一部分：canonical `routing_records.jsonl` 保持 P0 语义；sample rows 只在 `sample_routing_records.jsonl`（见 §3.1/§5）。
- 任何“v1 装不下”的需求（如需要 per-sample image/scene 字段）→ 停下请示，P1 内不得绕过。

## 7. measurement 要求（P1-B）

- 臂：OFF=仅 model forward；ON=P1-A 逐样本采集路径（每样本 forward + snapshot 刷新/force + adapter 生成记录 + 内存缓存；MoE 侧含 BN 状态恢复）；JSONL 写盘单独计（可选拆 serialization 子臂）。
- 参数（沿用 P0 脚本默认值，保守）：warmup=5，iterations=50/arm，size=640，device=cpu；独立 on/off 重复 ≥3 次。
- 报告：每对 overhead% + mean ± std + min/max + n；artifact 记录协议参数、model config、环境、时间戳、baseline 三元组（`3eb6cd9` / `d604c4b` / `aa5d2e2`）。
- MoE 的 snapshot 刷新与 BN 状态恢复开销必须显式计入 ON 臂并说明。
- 明确：不引用、不混入 P0 `overhead_result.json` 数值（P0 数字是历史证据，不是 P1 结果）。
- 阈值结论：P1-B 只报告统计量；不做 `<10%` 判定，除非另行预注册阈值与置信区间方法。

## 8. evidence 要求

- 目录沿用 `artifacts/smoke/<run_id>/`；P0 型 13 文件同名同语义保留。
- P1 新增证据：
  - `sample_routing_records.jsonl`（P1-A）：全行 v1、`step=int`；是 sample rows 的唯一落盘位置。
  - `sample_overhead_result.json`（P1-B）：协议元数据 + 统计量。
- manifest：`build_manifest()` 自动覆盖目录内除自身外全部文件；新文件必须在 manifest 生成前落盘（runner 收尾顺序已保证）。
- verify：`verify_manifest()` 语义不变；P1-A 新增对 sample 文件的校验（全量 `iter_records` 解析、run_id 一致、每 family `step` 连续 0..S_f-1、行数 = S_f × M_f、主键无重复）。
- 历史证据目录/文件一律不修改；P1 每次使用新 run_id。

## 9. acceptance criteria

P1-A（能力）：
1. 全量 pytest 通过（现有套件 + P1 新增单测）。
2. 一次真实 P1 smoke：P0 四 step 全部 PASS，新增 sample step PASS。
3. `sample_routing_records.jsonl` 满足 §8 校验（行数、step 连续、主键唯一、run_id 一致、全行 v1 可解析）。
4. `routing_records.jsonl` 仍为 P0 语义：只含 P0 四 step 产生的行（每 family 每模块一行、`step=None`），**不含任何 `step≠None` 的行**。
5. P1-A step 未把 sample rows 返回给 `execute_smoke_steps()`（即 summary 中该 step 无 `records` 计数，sample 行全部在独立文件）。
6. git diff 不含 `routing_record.py`、adapter、P0 产物、历史数字、上游代码改动。
7. MoE 逐样本单测断言 BN 隔离生效（乱序/重复 sample 的路由记录与顺序无关），或文档化声明采用 §3.3 兜底替代及代价。

P1-B（测量/验收）：
8. `sample_overhead_result.json` 存在且含协议元数据与统计量，数值有限，重复 ≥3 次。
9. manifest 覆盖新增文件，`--verify-artifacts` 返回 PASS。
10. 验收报告如实披露被测对象、参数、环境三元组；不引用 P0 overhead 数字；不做未预注册的阈值结论。

## 10. 不属于 P1（明确排除）

- 修改 `e3-routing/v1`、adapter `family_data` 白名单、`RoutingRecordWriter` 语义。
- 修改 P0 acceptance / P0 evidence / 历史 overhead 数字；删除或改变 P0 四 step。
- 实时面板、token 级原图热图、跨族“统一 schema”正式冻结。
- 上游 YOLO-Master 代码修改、上游 PR。
- 训练后 checkpoint 的坍塌复测、MoT 在 MOT 跟踪任务上的评估、训练减速 <10% 的正式结论。
- 数据集/样本量扩展（coco8 之外）、分布式/多卡（DDP）、`moa` 等未支持 family。
- 移除或替换 `_moe_force_snapshot` / `MOE_SNAPSHOT_INTERVAL` 机制（保留，仅按既有 known coupling 记录）。

## 11. 开放决策（遇此须停下请示）

- 任何需要改 v1 schema / adapter / writer 语义的需求。
- 样本集、数据集、真实 checkpoint、阈值与置信区间方法的引入。
- config 新增键；新 evidence 文件类型超出 §8 清单。
- §3.3 兜底替代若被采用，需先记录理由与代价并经确认。