# Routing Health Check（只读路由健康诊断）

目的：把既有的 MoE 路由坍缩（expert collapse）现象固化成可复现、可追溯的只读诊断——不训练、不改模型、不改 forward、不改上游、不改动任何既有 evidence。

## 1. 用法

```bat
python -m scripts.routing_health_check --run-dir artifacts/smoke/<run_id> --out artifacts/health/<run_id>/routing_health.json
```

- 输入：run 目录下的 `routing_records.jsonl`（canonical，`e3-routing/v1`）；`--records` 可改指同名的其它 JSONL。
- 输出：stdout 文本摘要（加 `--json` 输出机器可读 JSON）；`--out` 另存 JSON 诊断结果。
- 追溯：诊断 JSON 记录来源 `run_id`、`source` 路径与 `source_sha256`。

## 2. 指标与阈值（明确、可解释）

逐模块（family + layer）输出：Gini（`load_gini`）、归一化熵（`routing_entropy_normalized`）、top1_share（`dominant_expert_share`）、dominant expert（`dominant_expert`），以及 collapse 判定。

| 常量 | 值 | 判定 |
|---|---|---|
| `COLLAPSE_SHARE_THRESHOLD` | 0.80 | top1_share ≥ 0.80 → `collapse = true`（沿用 `docs/smoke-design-and-schema.md` §3 对 `collapse_flag` 的建议阈值 0.8） |
| `ONE_HOT_SHARE_THRESHOLD` + `ENTROPY_COLLAPSE_MAX` | 0.999 + 0.05 | top1_share ≥ 0.999 且 归一化熵 ≤ 0.05 → `severity = one_hot` |
| `UNIFORM_ENTROPY_MIN` + `UNIFORM_GINI_MAX` | 0.99 + 0.01 | 归一化熵 ≥ 0.99 且 Gini ≤ 0.01 → `uniform = true`（均匀锚点） |

- `severity`：`one_hot` → `concentrated`（0.80 ≤ top1_share < 0.999）→ `spread`。
- 同一模块存在多条记录时（如 sample JSONL）：Gini 与归一化熵取均值，top1_share 取**最坏（最大）**记录，dominant expert 取该记录；collapse 按最坏记录判定。
- 参考量 `gini_one_hot_reference = (E−1)/E`：单专家独占时 Gini 的闭式值；`gini_matches_one_hot` 表示实测 Gini 与之在 1e-6 内一致（见 `docs/p0-three-seed-evidence.md`）。

## 3. 非目标

- 只读取既有 artifacts，不重训、不改模型/forward/上游、不修改既有 evidence。
- 只报告路由结构性质（未训练快照的坍缩/均匀），不是性能提升或质量结论，不改变任何既有实验结论。
