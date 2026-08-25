# E3 准入任务：我的理解与执行记录

更新时间：2026-08-25（UTC+8）

本文是我对 E3 准入要求的个人理解与本次交付的说明，素材来自任务书、8.24 准入表与筹备材料。若导师后续发布新通知，以新通知为准。

## 1. 我对任务的理解

E3 的目标是给 YOLO-Master 的路由机制做一套可审计的观测链路：在不动核心 `forward` 的前提下，通过 hook 采集路由快照，输出结构化数据、字段定义和开销评估，最终支撑三族（MoE / MoT / Latent）统一 schema 的 P0 设计。

本次准入（Smoke 阶段）我按“能复现、能校验、边界诚实”三条原则交付：

- 能复现：一条命令跑完采集，配置、环境、日志、证据全部入库；
- 能校验：证据附 SHA-256 清单，指标有单测覆盖；
- 边界诚实：只声明完成了什么，不把 Smoke 结果冒充 P0/P1 结论。

## 2. 本次交付范围

| 交付物 | 说明 |
| --- | --- |
| 三族路由采集 | MoT（4 类合成场景）、MoE（coco8 真实图）、Latent（单张前向快照） |
| 结构化证据 | CSV / JSONL / PNG / summary.json / manifest.sha256.json |
| 字段字典草案 | 见 `docs/smoke-design-and-schema.md`，标注哪些字段是统一候选、哪些是族专属 |
| 开销实测 | 开关 hook 对照，实测数值见 `docs/overhead-and-risk-plan.md` |
| 单元测试 | 指标公式与校验逻辑的契约测试，`run_tests.cmd` |

明确不做（属于后续阶段）：正式冻结统一 schema、实时面板、token 级原图叠加、上游 PR、训练减速<10% 的正式结论。

## 3. 基线与环境

- 锁定基线 commit：`3eb6cd914b651a06e2cd08ea87d12c28cab95502`（2026-08-23，main 分支）。
- 运行环境：Python 3.11.9 + torch CPU（Windows 本机），具体版本快照见 `artifacts/smoke/admission-20260825/environment.json`。
- 数据：coco8（自动下载，约 433 KB）；MoT 用脚本内置合成场景，无需数据集。

## 4. 复现与成功判据

- 复现：`run_smoke.cmd`（采集）、`run_tests.cmd`（单测），详见根目录 README。
- 成功判据（全部满足才算 PASS）：
  1. 退出码 0，日志以 `result=PASS` 结尾；
  2. MoT 生成 4 个输出文件，CSV 覆盖 4 场景 × 3 专家；
  3. MoE 三个 router（`model.5/8/11.routing`）hook 成功，coco8 val 跑完，usage_stats 非空；
  4. Latent 三个模块（`model.23/24/25`）快照非空（约 35-36 字段，随机权重重跑允许 ±1 浮动）；
  5. 开销测量输出 `overhead: X.XX% (target < 10%)`；
  6. manifest 哈希可校验全部产物。

## 5. 已知限制与后续计划

- 随机初始化模型的坍塌现象（MoT `LocalConvTransformer` 恒占 100%）是训练前基线特征，后续要在真实 checkpoint 上复测。
- 三族字段粒度差异大，统一 schema 需要带 `routing_paradigm` 标记的父结构，不能简单拉平（见设计文档）。
- 后续按时间线推进：锁题 → 最小闭环 → 消融 → 中期演示 → 冻结可复现包 → 答辩。

## 6. 变更记录

- 2026-08-25：初版，完成三族 Smoke、开销实测与证据打包。