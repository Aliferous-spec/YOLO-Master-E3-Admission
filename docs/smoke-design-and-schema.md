# 三族路由观测设计与字段字典

## 1. 目标与边界

目标：在锁定版本相关代码上，对 MoT / MoE / Latent 三类路由各完成一次真实路由采集，使用 forward hook（或原生快照属性）获取结构化 routing 数据，并生成可审查的 CSV/JSONL/PNG 与完整证据包。

非目标：不训练模型、不评估 mAP、不冻结统一 schema、不证明训练减速<10%、不提交上游 PR。

## 2. 采集流程

```text
部署的 YOLO-Master baseline
  -> MoT: scripts/diagnose_mot_routing.py --synthetic (4 场景合成输入)
  -> MoE: ExpertUsageTracker + coco8 val (4 张真实图)
  -> Latent: 随机 640x640 单张前向，读取 last_routing_snapshot
  -> 结构化输出 -> artifacts/smoke/<run_id>/
  -> summary.json + manifest.sha256.json + full.log
```

hook 采集不替换模型输出；所有快照先 detach 再落盘，避免日志持有计算图。

## 3. 三族字段字典（Schema 草案，P0 核心交付物）

三类路由模块的原生字段粒度差异很大，统一 schema 需要字段对齐与降维，而不是简单拼接：

| 统一字段（拟） | MoE 来源 | MoT 来源 | Latent 来源 | 说明 |
|---|---|---|---|---|
| `num_experts` | `num_experts` | `num_experts` | `num_experts` | 三类原生都有，可直接对齐 |
| `top_k` | `top_k` | `top_k` | `top_k` / `training_top_k` / `inference_top_k` | Latent 区分训练/推理两套 top_k，需要在统一层做归一 |
| `expert_usage` | `ExpertUsageTracker.usage_stats` 聚合 hits/weighted_sum | `expert_usage`（直接张量） | `expert_usage` | 三类张量形状需统一为 `[num_experts]` 浮点列表 |
| `aux_loss` | `RoutingAuxPublisher` 统一通道 | `aux_loss` | `aux_loss` | 唯一天然已经跨三类统一的字段 |
| `dominant_expert` / `dominant_share` | MoE 诊断类原生支持 | 需从 `expert_usage` 现算 | 需从 `expert_usage` 现算 | MoT/Latent 缺此字段，需在采集层补算 |
| `collapse_flag` | 原生支持 | 需自定义阈值判断 | 需自定义阈值判断 | 建议阈值可先沿用 MoE 侧的 0.8 |
| `scene_context` | 无 | `scene_aware` / `scene_stats` / `scene_bias` | 无 | MoT 独有字段，其余两类留空 |
| `value_fusion_*` | 无 | 无 | `value_fusion_mode` / `value_fusion_weights` | Latent 独有字段，体现其“融合”而非“选择”的路由范式 |

核心难点：MoE 偏向“离散选择”语义（dominant expert、collapse），MoT 带场景条件（scene-aware），Latent 是“连续融合”语义（value fusion），三者不是同一套路由范式的简单变体。统一 schema 需要设计带 `routing_paradigm` 标记位的父结构，而不是强行拉平字段。

## 4. 指标定义

令非负负载 `u_i`，`p_i = u_i / Σu_i`，专家数为 `E`：

- 路由熵：`H = -Σ p_i ln p_i`；归一化熵 `H / ln(E)`。
- 主导专家占比：负载份额的最大值 `max_i p_i`。
- Gini：对负载升序 `x_(i)`，`G = 2Σ i·x_(i)/(E·Σx) - (E+1)/E`，裁剪到 `[0,1]`。

单测覆盖均匀 `[1,1,1]`（归一化熵=1、Gini=0）与极端 `[1,0,0]`（归一化熵=0、Gini=2/3）两个已知分布。

## 5. 通过标准

1. 命令退出码 0，日志以 `result=PASS` 结束；
2. MoT：`[routing] wrote ...` 出现 4 次，CSV 覆盖 4 场景 × 3 专家；
3. MoE：`model.5/8/11.routing` 均成功挂 hook，`usage_stats` 非空且数值正常，coco8 val 跑完；
4. Latent：`model.23/24/25` 均输出非空 `last_routing_snapshot`（约 35-36 个字段，随机初始化重跑可能有 ±1 浮动，属正常）；
5. 开销测量输出 `overhead: X.XX% (target < 10%)`；
6. JSONL/CSV、PNG、config、environment、summary、full.log 与 manifest 同时存在，哈希可校验。

## 6. 证据图说明

- `mot_expert_heatmap_top1_share.png`：4 场景 × 3 专家的 `top1_share` 热力图，逐格显示占比。
- 随机初始化模型在全部 4 类合成场景下 `LocalConvTransformer` 专家 `top1_share` 恒为 1.00（专家坍塌），这是训练前基线的真实特征，不是脚本 bug；后续需在真实训练 checkpoint 上复测。

## 7. 已知耦合（known coupling）

- **MoE 采集依赖上游模块私有属性 `_moe_force_snapshot`**：上游 MoE router 默认按 `MOE_SNAPSHOT_INTERVAL` 间隔才刷新 `last_routing_snapshot`；为在验收所用的真实 forward 上拿到快照，采集代码对 MoE 模块直接设置上游私有属性 `_moe_force_snapshot = True`（`scripts/run_e3_smoke.py` 的 `_capture_model_records(force_moe_snapshot=True)` 与 `run_moe()` 各设置一次）。这是对上游未公开接口的实现耦合；本仓库不新增接口、不封装新功能，当前保留该机制并仅在此记录依赖。
- 升级/切换 YOLO-Master 基线版本时需回归验证：若上游改名或删除 `_moe_force_snapshot`，MoE 快照采集会静默回退为按间隔刷新或不再产出，导致 MoE v1 记录缺失（现有失败隔离会把该 step 标记 FAIL 并保留日志）。
- 另见：`docs/requirements.md` §5 已知限制、README「版本与边界」。
