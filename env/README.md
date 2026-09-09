# E3 准入包：复现环境说明

本包涉及**两套**环境，用途不同，必须区分，不可混用声明。

### A. 证据环境（产出 artifacts 的验收环境，权威）

每次 run 的 `artifacts/smoke/<run_id>/environment.json` 为运行时自动采集的
权威快照。本仓库全部归档 run（含 P0 三 seed）一致：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.9 |
| PyTorch | 2.13.0+cpu |
| ultralytics（YOLO-Master 部署基线） | 8.4.101 |
| 平台 | Windows-10-10.0.26200-SP0 |

代码中的 `_MOE_SAMPLE_VALIDATED_ULTRALYTICS = "8.4.101"` 是"已验证版本
标注"，不是硬版本锁：真实门槛为 `>= 8.4.0`（见
`scripts/run_e3_smoke.py:_require_moe_validator_api`）。

### B. 最小单元测试环境（仅供跑 pytest，不产出证据）

| 组件 | 版本 |
| --- | --- |
| Python | 3.9（CI 为 Ubuntu + Python 3.9） |
| PyTorch | CPU 版（任意近期版本） |
| ultralytics | 由 CI 按基线 commit `3eb6cd9` 安装 |

单元测试只依赖指标与 schema 逻辑，可在该最小环境独立运行；完整 smoke 采集
必须在环境 A 下执行，还需要 YOLO-Master 部署目录与权重/数据，见根目录 README。

## 版本声明纪律

- 任何版本声明（README / docs / 本文档 / environment.json）必须与**实际执行
  环境**一致；不得沿用旧声明或伪造未运行过的版本。
- 每次 smoke run 的 `artifacts/smoke/<run_id>/environment.json` 是当次运行的
  权威环境快照（运行时自动采集），以它为准。
- 若换环境重跑，请如实更新当次 environment.json，而不是改写历史数字。

## 最小单元测试环境（与 CI 一致）

```bash
python -m venv .venv            # Python 3.9
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install pytest pyyaml
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install "ultralytics @ git+https://github.com/Tencent/YOLO-Master.git@3eb6cd914b651a06e2cd08ea87d12c28cab95502"
.venv/bin/python -m pytest tests -q
```

CI 定义见 `.github/workflows/ci.yml`；Windows 本机一键入口为 `run_tests.cmd`。
