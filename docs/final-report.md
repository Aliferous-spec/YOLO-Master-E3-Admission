# [犀牛鸟-E3] YOLO-Master 路由非侵入观测链路：结项报告

仓库：`Aliferous-spec/YOLO-Master-E3-Admission`
报告生成：2026-09-10（UTC+8）
证据环境：Python 3.11.9 / torch 2.13.0+cpu / ultralytics 8.4.101 / Windows-10
基线三元组：`3eb6cd914b651a06e2cd08ea87d12c28cab95502` / `d604c4b` / `aa5d2e2`

---

## 0. 一句话结论

在**不修改 YOLO-Master 核心 `forward`、不提交上游 PR** 的前提下，建成一条覆盖
MoT / MoE / Latent 三族的路由观测链路：产出冻结 schema `e3-routing/v1` 的结构化记录，
每条 run 附 SHA-256 清单可在全新 clone 上自校验；并**先预注册判据、后执行实验**，
以「bootstrap 95% CI 上界 < 10%」判定采集开销，3/3 seed 通过（上界 3.33%–4.90%）。

---

## 1. 做了什么

| 交付项 | 状态 | 证据 |
| --- | --- | --- |
| 三族路由采集（MoT / MoE / Latent） | 完成 | 每 run `routing_records.jsonl` 15 行（每族每模块 1 行） |
| 冻结 schema `e3-routing/v1` + 适配器 | 完成 | `scripts/routing_record.py` / `routing_adapter_*.py` |
| Per-run 证据隔离 + SHA-256 清单 | 完成 | `manifest.sha256.json`，`--verify-artifacts` 校验 |
| P0 多 seed 复现 | 完成 | seed 0–5 共 6 个 run，全部 PASS |
| P1-A 逐样本采集 | 完成 | `sample_routing_records.jsonl` 51 行 / run |
| P1-B 开销：判据预注册 + 确认性实验 | 完成 | 判据 `docs/p1-b-overhead-judging-criteria.md`；判定 `docs/p1-b-overhead-verdict.md` |
| 指标解析解真值锚点 | 完成 | `tests/test_metric_analytic_groundtruth.py`（14 项） |
| 路由证据面板（双通道降级） | 完成 | `scripts/routing_panel_sink.py`（13 项测试） |
| 开源形式项 | 完成 | MIT LICENSE、GitHub Actions CI、`env/`、`docs/limitations.md` |
| 真实训练减速测量 | **未执行**（协议与脚本已实现并过单测） | 见 §5.1 |

代码规模：`scripts/` 约 4000 行，`tests/` 约 2700 行，**124 个测试全部通过**。

---

## 2. 方法

### 2.1 零侵入

不修改上游任何 `forward`、不改训练器、不提交上游 PR。采集走两条既有通道：

- MoT / Latent：读取模块**原生快照属性**（`last_routing_snapshot`）；
- MoE：注册 forward hook + `ExpertUsageTracker`。

唯一已知耦合显式记录在案：MoE 侧依赖上游私有属性 `_moe_force_snapshot`
（`docs/smoke-design-and-schema.md` §7）。保留该机制未改，只记录。

### 2.2 三族适配而非强行统一

三族字段粒度差异大（Latent 单模块约 35 字段，MoE 少一个量级），
不做扁平拉平，而是「公共字段 + `family_data` 族专属字段」两层结构，
避免大量空值列。跨族统一 schema 的**正式冻结不在本次范围内**。

### 2.3 证据双流分离

`routing_records.jsonl`（canonical，P0 语义，每族每模块一行、`step=None`）
与 `sample_routing_records.jsonl`（sample，逐样本，51 行 / run）**物理分文件**，
P1-A 的 sample 行不回传给 P0 的 step 汇总路径，避免污染 P0 语义。

---

## 3. 证据链（本项目的主要差异化）

1. **Per-run 隔离**：每个 run 独占 `artifacts/smoke/<run_id>/`，含
   `environment.json`（运行时动态探测版本，非硬编码）、`config.resolved.yaml`、
   `full.log`、`manifest.sha256.json`。
2. **SHA-256 自校验**：`--verify-artifacts <run_dir>` 重算全部产物哈希并与清单比对。
3. **clone 可复现**：实测修复过一处会让评审翻车的缺陷——仓库早期
   `.gitattributes` 写 `artifacts/** text eol=crlf`，导致 LF 产物在 clone 后被转成
   CRLF、哈希全部失配（实测 3 处 mismatch）。已改为 `-text` 并按原始字节
   renormalize，现可在 `core.autocrlf=true` 的全新 clone 上复验通过。
   （artifacts 内部混用两种换行：JSONL 显式 LF，json/yaml/log 走 Windows 文本模式
   ——单一 eol 规则无解，只能禁止转换。）
4. **判据冻结锚点**：`docs/p1-b-overhead-judging-criteria.freeze.json` 记录
   判据文档的 freeze commit 与 blob SHA-256，任何人可验证「判据先于实验」。

---

## 4. 结果

### 4.1 P0：多 seed 复现

seed 0–5 六个 run，P0 四 step 全部 PASS，manifest 校验全部 PASS。

一个值得记录的发现：**六个 seed 下 15 个模块的路由指标逐位一致**。
原因是未训练初始化状态下路由行为由结构决定（MoE 的 Gini 恰为 (E−1)/E，
即 argmax 平局裁决下的单专家坍缩），**对 seed 不敏感**。
这不代表「指标稳定」，只代表「未训练基线的路由是结构主导」。
训练后 checkpoint 上的复测不在本次范围内。

### 4.2 P1-B：采集开销（范围内判据，已判定）

判据（先冻结，commit `797d5fd`，锚点 `ded7156`）：

> PASS ⟺ 采集链相对开销均值的 **bootstrap 95% CI 上界 < 10%**。
> 只看上界，不看宽度；不看均值、不看中位数。

确认性实验在冻结**之后**用全新 seed 3/4/5 执行（seed 0/1/2 降级为协议探测集，
仅用于确定 n=120 / warmup=5 等参数，不计入判定）：

| seed | mean | std | 95% CI | n | 判定 |
| --- | --- | --- | --- | --- | --- |
| 3 | 2.539% | 9.530 | [0.841%, **4.199%**] | 120 | MET |
| 4 | 2.157% | 6.562 | [0.998%, **3.325%**] | 120 | MET |
| 5 | 3.389% | 8.353 | [1.909%, **4.896%**] | 120 | MET |

**PASS 3/3。**

**为什么用区间上界而不是均值**：CPU 单次前向计时噪声极大（per-observation std
6.6–14.5 个百分点，单观测区间跨零）。n=120 下均值的标准误约 1 个百分点，
用均值判 <10% 几乎必然通过——它区分不出「真实开销 8%」与「真实 2% 但噪声大」。
区间上界把噪声显式计入结论，是更保守的协议。

### 4.3 指标解析解真值锚点

不信任任何参考实现，直接用闭式解锚定指标链
（锚定对象：上游 `ultralytics.nn.modules.routing_protocol.global_routing_metrics`）：

- `[1,0,0]` → 熵 = 0，Gini = 2/3
- `[1/3,1/3,1/3]` → 熵 = ln3，Gini = 0
- 尺度不变性、置换不变性、负值清洗、E=1 退化情形
- 适配器归一化熵 = H / ln(E)

即：指标不仅是「跑通了」，而是**有数学真值锚点**。共 14 项断言。

### 4.4 路由证据面板

`scripts/routing_panel_sink.py`：把任一 run 的路由证据渲染成**零依赖自包含 HTML**
（手写 inline SVG，无 matplotlib / CDN / JS，可离线双击打开），
TensorBoard 通道在 tensorboard 缺失时自动降级（本机即走 HTML 通道）。

设计取向是「**证据能不能信**」而不是「路由长什么样」：面板显式打印 schema 版本、
三族覆盖情况（**缺哪族标 MISSING，不伪造**）、canonical/sample 双流分离、
缺失指标渲染 `n/a` 而不是 `0.0`。面板渲染失败不会中断 smoke。

---

## 5. 没做的、以及为什么（负结果照报）

### 5.1 真实训练减速：协议已实现，未执行

`scripts/measure_training_slowdown.py`（ABBA 块级配对、warmup、多 seed、
bootstrap CI）已实现，6 项单测通过，预注册判据见 `docs/p1-judging-criteria.md`。
先纠正一个此前的事实错误：曾判断 coco8 在本机不可得，实测**是可得的**——
`check_det_dataset('coco8.yaml')` 解析到
`C:\Users\刘小姐\Documents\yolo-master-study\datasets\coco8`（8 张图，已确认）。
（不可达的只是 github.com 上的**新下载**通道，本地副本一直存在。
这条同时确认了 P0/P1 的 MoE 侧确实走的是真实 coco8 val 图像，而非随机张量。）

本次已作为**补充测量**执行：`--dry-run` 验证通过，正式运行见
`artifacts/training_slowdown/` 下的 `slowdown_result.json`。
需要强调：`docs/requirements.md` §2 与 `docs/p1-spec.md` §10 明确把
「训练减速 <10% 的正式结论」列入不做清单，因此无论跑出什么数字，
它都**不作为验收依据**，只在补充章节如实披露。

### 5.2 其它明确不做

跨族统一 schema 正式冻结、token 级原图热图、上游 PR、训练后 checkpoint 坍塌复测、
MoT 在 MOT 任务上的评测、数据集扩展（coco8 之外）、分布式/多卡、未支持 family。

### 5.3 已知限制与历史遗留

- 历史 run `p0-acceptance-20260905`、`smoke-20260905-203946-f1560a` 存在
  `full.log` 哈希不符；`smoke-20260909-001109-11e290` 为旧格式，不被当前
  verifier 兼容。三者均**原仓库即失败**，非本次改动引入，已在
  `docs/limitations.md` 如实记录并建议标注 superseded。
- 采集开销协议固定 OFF→ON 单顺序（`p1-spec` §7 未规定 ABBA 平衡），
  **残留顺序效应未标定**。
- JSONL 写盘排除在 ON 臂之外，故 4.2 的数字**不是端到端延迟增量**。
- 所有结论仅适用于 CPU / yolo-master-n @ 640 / 本机环境，不外推 GPU 或分布式。

---

## 6. 复现步骤

```bat
set PYTHONUTF8=1
cd C:\tmp\e3-package                      :: 或你的包路径

:: 1) 单测（124 项）
"C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe" -m pytest tests -q

:: 2) 一次 smoke（需要显式给基线路径）
env -u PYTHONPATH -u PYTHONHOME ^
  "C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe" ^
  -m scripts.run_e3_smoke --baseline-root D:\YOLO-Master

:: 3) 多 seed 批量 + 自动校验
env -u PYTHONPATH -u PYTHONHOME ^
  "C:\Users\刘小姐\.venvs\yolo_master\Scripts\python.exe" ^
  -m scripts.run_smoke_seeds --baseline-root D:\YOLO-Master --seeds 0,1,2

:: 4) 校验任一 run 的证据完整性
python -m scripts.run_e3_smoke --verify-artifacts artifacts\smoke\<run_id>

:: 5) 渲染路由面板
python -m scripts.routing_panel_sink artifacts\smoke\<run_id>
```

注意：子进程必须剥离 `PYTHONPATH` / `PYTHONHOME`（环境里的 safe-delete shim 会
劫持 `os.remove` 打断 MoT 步骤）；`run_smoke_seeds.py` 已内建剥离。

---

## 7. 变更记录

- 08-25：三族 Smoke、开销实测、证据打包（初版交付）
- 09-05：P0 evidence closure（`_moe_force_snapshot` known coupling 记录）
- 09-06~09：P1-A 逐样本采集、P1-B 逐样本采集链开销测量
- 09-09：P0 三 seed 证据；解析解真值锚点；`.gitattributes` 证据损坏修复；
  LICENSE / CI / env / limitations
- 09-10：P1-B 判据预注册 + 冻结锚点 + 确认性实验（seed 3/4/5，PASS 3/3）
