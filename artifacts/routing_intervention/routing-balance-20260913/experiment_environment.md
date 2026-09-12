# Routing balance intervention: environment and run provenance

Experiment id: `routing-balance-20260913`. One real training session per arm/seed,
6 sessions total, executed 2026-09-13T01:21:48+08:00 .. 2026-09-13T01:26:47+08:00.

## Checkouts

| role | path | commit | branch | status |
| --- | --- | --- | --- | --- |
| baseline (trained) | `D:\YOLO-Master` | `aa5d2e20c109b96f4a0c68f667ed2694586ef745` | `baseline/2026-08-22` | clean before and after |
| observation harness | `C:\tmp\e3-package` | `9e61f18cce3b635a2218267cadb0207e4f44a7df` | `e3-final-clean` | untracked experiment scripts only |

No file under `D:\YOLO-Master` was created or modified: the intervention is a plain
Python attribute on the deployed model.

## Environment prerequisites

- `POLARS_SKIP_CPU_CHECK=1` is required before training starts; without it the trainer
  aborts with a misleading non-finite-state error. Recorded per run as
  `environment.polars_skip_cpu_check = "1"`.
- `PYTHONPATH=D:\YOLO-Master` (recorded per run as `environment.pythonpath`).
- Platform `Windows-10-10.0.26200-SP0`, Python 3.11.9, torch 2.13.0+cpu,
  ultralytics 8.4.101 (resolved from `D:\YOLO-Master\ultralytics\__init__.py`).

## Protocol (pre-registered, identical for both arms)

- arms: `baseline` = `balance_loss_coeff=1.0`, `intervention` = `moe_loss_fn.balance_loss_coeff=4.0`
- seeds: 0, 1, 2; 12 epochs each = 2 warmup + 10 measured
- data `ultralytics/cfg/datasets/coco8.yaml`, model
  `ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml`, imgsz 640, batch 1,
  device cpu, workers 0, deterministic, patience 13
- both arms use the same unmodified `scripts.routing_capture` chain and force
  `_moe_force_snapshot` on all 3 routed blocks
- measured per epoch per target layer: Gini, normalized entropy, top-1 share, dominant
  expert, module aux loss, model `_last_mixture_aux_loss`, epoch wall-clock
- coefficient asserted at `on_train_start`, every `on_train_epoch_start` and `on_train_end`

Target layers: `model.5` (4 experts, Gini ceiling 0.7500), `model.8` (8 experts, 0.8750),
`model.11` (16 experts, 0.9375).

## Per-run timing

| run | started | finished | wall s |
| --- | --- | --- | --- |
| seed0_baseline | 01:21:48 | 01:22:17 | 28.76 |
| seed1_baseline | 01:22:19 | 01:22:48 | 29.05 |
| seed2_baseline | 01:22:50 | 01:23:36 | 46.19 |
| seed0_intervention | 01:23:44 | 01:24:19 | 35.05 |
| seed1_intervention | 01:24:21 | 01:25:35 | 74.60 |
| seed2_intervention | 01:25:38 | 01:26:47 | 69.77 |

Total process wall-clock 283.41 s; measured-epoch compute 108.95 s.

## Integrity

- 6/6 runs present, no missing seed or arm (`comparison.json` -> `integrity.problems = []`)
- 36 routing records and 30 measured metric rows per run, 180 rows total
- 14/14 coefficient assertions green in every run; no NaN, no non-finite observation, no
  callback overwrite detected in any run

## Result and caveat

Verdict is NOT PASS (null): mean paired layer-Gini delta +0.000622 (95% CI
[0.000000, +0.001865]), normalized entropy delta 0.002756 (95% CI [-0.008268, 0.000000]).

Caveat: 143 of the 180 measured rows sit exactly on their layer Gini ceiling
`(n-1)/n` with normalized entropy 0 and top-1 share 1.0, i.e. fully collapsed routing.
Seeds 1 and 2 are at the ceiling on all 3 layers in both arms (paired delta exactly 0);
the whole paired delta comes from seed 0, where the intervention moved Gini slightly up
and entropy slightly down. The observable therefore had almost no headroom, and the
4x balance coefficient did not rebalance routing on this workload.