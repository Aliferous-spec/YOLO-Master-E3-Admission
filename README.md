# E3 路由透视镜：三族准入 Smoke（MoT / MoE / Latent）

Owner：刘欣燃（GitHub：`Aliferous-spec`）

状态：**PASS — 完成 MoT / MoE / Latent 三族准入 Smoke + 开销实测，不宣称完成 P0。**

本仓库是 E3 准入审核包：在已部署的 YOLO-Master 上，以非侵入 forward hook / 原生快照属性对 MoT、MoE、Latent 三类路由各采集一次 routing 快照，提供结构化日志、CSV/JSONL/PNG 证据、字段字典、开销测量结果与风险降级。没有修改 YOLO-Master 核心 `forward`，没有提交上游 PR。

## 交付清单对照

| 准入项 | 材料 |
| --- | --- |
| 环境安装 | `artifacts/smoke/admission-20260825/environment.json` |
| 基线与范围 | README「版本与边界」+ `docs/requirements.md` |
| 复现 | `run_smoke.cmd`（采集）/ `run_tests.cmd`（单测） |
| 配置文件 | `configs/e3_smoke.yaml` 与 `artifacts/.../config.resolved.yaml` |
| 完整日志 | `artifacts/smoke/admission-20260825/full.log` |
| 结果证据 | `summary.json`、`mot_routing_*.csv`、`mot_expert_heatmap_top1_share.png`、`moe_usage_stats.json`、`latent_snapshot.jsonl`、`overhead_result.json`、`manifest.sha256.json` |
| 字段字典 | `docs/smoke-design-and-schema.md` |
| 开销与降级 | `docs/overhead-and-risk-plan.md` |

## 一键复现

本仓库与已部署的 `YOLO-Master` 目录同级。在 Windows CMD 中执行：

```bat
cd /d "C:\path\to\YOLO-Master-E3-Admission"
set BASELINE_PY=C:\path\YOLO-Master\.venv\Scripts\python.exe
run_smoke.cmd
```

单元测试：

```bat
run_tests.cmd
```

脚本自动使用部署环境；首次运行若本地没有 coco8，会下载约 433 KB 数据。也可以直接传 `--baseline-root` 或设置 `BASELINE_ROOT` 环境变量指定 YOLO-Master 路径。

## 实测结果（2026-08-25 本机复跑）

- **MoT**：官方合成脚本，4 类场景（dense_small / large_regular / irregular_occluded / sparse_small）× 3 专家，输出 `mot_routing_detailed.csv` / `mot_routing_scenarios.csv` / `mot_deformable_activation_check.csv` / `mot_expert_heatmap_top1_share.png`。
- **MoE**：`ExpertUsageTracker` 在 coco8 真实验证集（4 张图）上对 `model.5/8/11.routing` 三个 router 成功采集 hits / weighted_sum。
- **Latent**：`model.23/24/25` 三个 LatentMixture 模块输出非空 `last_routing_snapshot`，字段数约 35-36。
- **开销**：yolo-master-n @ 640×640，50 次前向，on/off 对照实测 1.50%~2.55%（运行间有波动，以 `overhead_result.json` 为准），满足 < 10% 目标。
- **已知基线现象**：随机初始化模型在全部 4 类合成场景下 `LocalConvTransformer` 专家 `top1_share` 恒为 1.00（专家坍塌），属训练前基线真实特征，不用于判断训练后质量。

![E3 准入 Smoke 静态图](artifacts/smoke/admission-20260825/mot_expert_heatmap_top1_share.png)

## 版本与边界

- 锁定基线：`3eb6cd914b651a06e2cd08ea87d12c28cab95502`（2026-08-23，main 分支）。
- schema 当前标注为 `e3-routing-smoke/v0.1-candidate`，仅作准入候选，不代表 P0 三族统一 schema 已冻结。
- 当前覆盖 MoT / MoE / Latent 三族；实时面板、token 原图热图和正式统一 schema 属于后续阶段。