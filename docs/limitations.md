# E3 准入包：历史失败 run 记录与边界

> 如实记录，不伪装。本文记录本项目开发/验收期间出现过的 3 个失败或作废的
> run。它们均为历史遗留（historical），已分别被后续 PASS run 取代
> （superseded），不代表当前版本状态；不得把其中任何一个当作当前版本通过。

## 1. 历史 run 失败清单

| # | run / 事件 | 时间 | 现象 | 根因 | 取代（superseded by） |
| --- | --- | --- | --- | --- | --- |
| 1 | `p0-acceptance-20260905`（首次执行） | 2026-09-05 18:05 | `moe step failed: MoE: no last_routing_snapshot captured after val forward`；`result=FAIL` | 开发期 MoE 采集链路在首次 val forward 后未取到快照 | 同日 18:20 同 run_id 重跑 PASS；仓库现存 `artifacts/smoke/p0-acceptance-20260905/` 为该 PASS 版 |
| 2 | `smoke-20260905-203946-f1560a` 的 `--verify-artifacts` 复核 | 2026-09-05 | `sha256 mismatch: full.log`；`result=FAIL (1 error)` | git 文本规范化改写归档证据导致哈希失配；由 `4947605` / `d4c9ea5`（stop git from corrupting hash-pinned evidence / store artifact blobs as raw bytes）修复 | `smoke-20260905-204546-6c7389`（P0 acceptance run，manifest verify PASS） |
| 3 | `smoke-20260909-001109-11e290`（P1-B 早期逐样本开销 run） | 2026-09-09 00:11 | 整块 50 轮 OFF→ON 计时（n=3），测量协议不达标，不构成 P1-B 验收证据 | 早期“整块”协议把慢漂移计入差值，已被取代；`scripts/measure_sample_capture_overhead.py` 文档已标注该设计为 superseded，`docs/p1-spec.md` §7 采用迭代级交替配对 | `smoke-20260909-003356-2960be`（n=120 配对协议，`result=PASS`，P1-B 验收 run） |

## 2. 当前状态与边界

- 仓库内现存归档 run 的 `summary.json` 均为 PASS；上述失败均为历史遗留，
  不伪装成当前版本通过。
- 本仓库此前未配置 CI；自 `.github/workflows/ci.yml` 起，pytest 通过/失败
  以 CI 退出码为准（失败非 0）。
- 以上失败的根因均为本包开发期实现或仓库卫生问题，不涉及 YOLO-Master
  核心 `forward` 与路由逻辑；本包亦未修改 YOLO-Master 核心代码。
- 清单中 #1、#2 的原始失败日志保存在开发机本地（未入库），#3 的作废产物
  仍保留在 `artifacts/smoke/smoke-20260909-001109-11e290/` 供核对。
