#!/usr/bin/env python3
"""Measure the E3 training slowdown (observation chain ON vs OFF) with real training.

Pre-registered protocol (``docs/p1-judging-criteria.md``, frozen; this script
never edits the criteria):

- Real training only: each seed runs one real ``YOLO(...).train(...)`` session on
  the baseline MoE model + coco8 workload; a bare forward is never a substitute.
- Seeds: 0 / 1 / 2 by default (one session per seed).
- ABBA block pairing per seed: after ``--warmup-epochs`` unmeasured prefix
  epochs, the session runs four contiguous blocks of ``--epochs-per-block``
  epochs in OFF, ON, ON, OFF order.  Pairing adjacent OFF/ON blocks (blocks 1-2
  and 3-4) suppresses slow machine drift inside the seed.
- ON and OFF blocks are the same workload: same model init (per-seed
  ``torch.manual_seed``), same data / batch / per-epoch step count; the only
  difference is the observation chain toggle.
- Reported statistics: point estimate (mean paired ``slowdown_percent``) plus a
  percentile bootstrap 95% CI.  The unique pre-registered pass criterion is
  CI upper bound < 10%.

Measured unit (block seconds): sum over the block's epochs of the per-epoch
batch-loop wall time taken between the trainer's ``on_train_epoch_start`` and
``on_train_epoch_end`` callbacks.  Per-epoch validation, EMA and checkpoint
bookkeeping run outside the window and are identical in both arms.

ON observation chain (per training batch of an ON block; e3-side only, no core
trainer / routing code is modified): routed MoE modules are set to refresh
``last_routing_snapshot`` on every forward (``_moe_force_snapshot = True``, the
convention the P1-A sample chain already uses), then the existing
``scripts.routing_capture.capture_records`` discovers and adapts each snapshot
into a v1 ``RoutingRecord``, which ``scripts.routing_record_writer`` appends to
the seed's ``seed<seed>_on_records.jsonl``.  Serializing and writing the JSONL
is part of the measured ON chain.

Dry run: ``--dry-run`` validates arguments, baseline / model / data paths,
seeds, the ABBA block layout, the ON chain wiring and prints the output schema
without building a model or training anything.

Example (formal run on the deployed baseline):
    <baseline-python> scripts/measure_training_slowdown.py \
        --baseline-root C:/path/to/YOLO-Master --run-id train-slowdown-<stamp>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Same workload defaults as the E3 smoke / P1-B overhead steps.
DEFAULT_MODEL = "ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml"
DEFAULT_DATA = "ultralytics/cfg/datasets/coco8.yaml"
# Frozen pre-registration (docs/p1-judging-criteria.md section 2).
ABBA_ORDER = ("OFF", "ON", "ON", "OFF")
PASS_CI_UPPER_PERCENT = 10.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

SCHEMA_VERSION = "e3-training-slowdown/v1"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_cfg_path(baseline_root: Path, value: str) -> Path:
    """Absolute path for a model/data cfg: absolute or relative to baseline_root."""
    path = Path(value)
    return path if path.is_absolute() else (baseline_root / path).resolve()


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError(f"--seeds must be a comma-separated list, got {raw!r}")
    return seeds


def _block_layout(epochs_per_block: int, warmup_epochs: int) -> list[dict[str, Any]]:
    """Return the four ABBA blocks with 0-based trainer epoch ranges."""
    if epochs_per_block < 1:
        raise ValueError(f"epochs_per_block must be >= 1, got {epochs_per_block}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    layout = []
    for index, arm in enumerate(ABBA_ORDER):
        layout.append(
            {
                "block": index + 1,
                "arm": arm,
                "epoch_start": warmup_epochs + index * epochs_per_block,
                "epoch_end": warmup_epochs + (index + 1) * epochs_per_block,
            }
        )
    return layout


def _slowdown_percent(off_seconds: float, on_seconds: float) -> float:
    """Paired ON/OFF slowdown in percent of the OFF arm time."""
    off = float(off_seconds)
    on = float(on_seconds)
    if not (math.isfinite(off) and math.isfinite(on)) or off <= 0.0:
        raise ValueError("arm timings must be finite and off_seconds > 0")
    value = (on - off) / off * 100.0
    if not math.isfinite(value):
        raise ValueError("slowdown_percent must be finite")
    return value


def _pair_rows(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turn per-seed ABBA block rows into adjacent-pair slowdown observations.

    Block order inside a seed must be ABBA_ORDER (OFF, ON, ON, OFF).  Adjacent
    pairs are blocks 1-2 and blocks 3-4 (ON normalized by the paired OFF block);
    using adjacent blocks suppresses slow drift within the seed.
    """
    seeds = sorted({int(row["seed"]) for row in blocks})
    pairs: list[dict[str, Any]] = []
    for seed in seeds:
        per_block = {int(row["block"]): row for row in blocks if int(row["seed"]) == seed}
        if set(per_block) != {1, 2, 3, 4}:
            raise ValueError(f"seed {seed}: expected blocks 1..4, got {sorted(per_block)}")
        arms = tuple(per_block[block]["arm"] for block in (1, 2, 3, 4))
        if arms != ABBA_ORDER:
            raise ValueError(f"seed {seed}: expected ABBA order {ABBA_ORDER}, got {arms}")
        for pair_number, (off_block, on_block) in ((1, (1, 2)), (2, (4, 3))):
            off_seconds = float(per_block[off_block]["seconds"])
            on_seconds = float(per_block[on_block]["seconds"])
            pairs.append(
                {
                    "seed": seed,
                    "pair": pair_number,
                    "off_block": off_block,
                    "on_block": on_block,
                    "off_seconds": off_seconds,
                    "on_seconds": on_seconds,
                    "slowdown_percent": _slowdown_percent(off_seconds, on_seconds),
                }
            )
    return pairs


def compose_payload(
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    model_config: str,
    data: str,
    seeds: Sequence[int],
    epochs_per_block: int,
    warmup_epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    blocks: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Assemble the training-slowdown artifact payload from measured blocks."""
    from scripts.measure_sample_capture_overhead import (
        _BASELINE_ROOT_HEAD,
        _OFFICIAL_BASE_REF,
        _ULTRALYTICS_EDITABLE_HEAD,
        _bootstrap_ci_mean,
        _distribution,
    )

    if len(blocks) != len(seeds) * len(ABBA_ORDER):
        raise ValueError(f"blocks {len(blocks)} != seeds {len(seeds)} x {len(ABBA_ORDER)}")
    if {int(row["seed"]) for row in blocks} != {int(seed) for seed in seeds}:
        raise ValueError("block seeds do not match the configured seeds")
    pairs = _pair_rows(blocks)
    values = [float(pair["slowdown_percent"]) for pair in pairs]
    stats = _distribution(values)
    stats["ci95"] = _bootstrap_ci_mean(values, resamples=bootstrap_resamples, seed=bootstrap_seed)
    upper = float(stats["ci95"][1])
    verdict = {
        "criterion": f"95% CI upper bound < {PASS_CI_UPPER_PERCENT}% (unique pre-registered pass criterion)",
        "ci95_upper_percent": upper,
        "pass": bool(upper < PASS_CI_UPPER_PERCENT),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "protocol": {
            "name": "e3-training-slowdown",
            "criteria_ref": "docs/p1-judging-criteria.md",
            "design": (
                "real YOLO training; per seed one session; ABBA epoch blocks "
                f"({', '.join(ABBA_ORDER)}) of {epochs_per_block} epochs after "
                f"{warmup_epochs} unmeasured warmup epochs"
            ),
            "measured_object": (
                "training wall-time slowdown of the e3 routing observation chain: "
                "ON arm runs the chain (MoE _moe_force_snapshot -> "
                "routing_capture.capture_records discovery/adapters -> "
                "RoutingRecordWriter JSONL append) once per training batch; "
                "OFF arm runs the identical training loop without it"
            ),
            "arms": {
                "off": "plain YOLO training batch loop",
                "on": "same batch loop + one e3 observation-chain pass per batch end",
            },
            "pairing": {
                "order": "OFF, ON, ON, OFF per seed; adjacent pairs (block1, block2) and (block4, block3)",
                "rationale": "adjacent ON/OFF block pairing suppresses slow machine drift within a seed",
            },
            "block_unit": (
                "seconds; per block = sum of per-epoch batch-loop wall time between the "
                "trainer on_train_epoch_start and on_train_epoch_end callbacks; per-epoch "
                "validation / EMA / checkpoint bookkeeping is excluded and identical in both arms"
            ),
            "same_workload": (
                "ON and OFF blocks train the same model init (per-seed torch.manual_seed), "
                "same data, same batch, same per-epoch step count; only the observation "
                "chain differs"
            ),
            "device": device,
            "threshold": f"95% CI upper bound < {PASS_CI_UPPER_PERCENT}% (only pass criterion)",
        },
        "parameters": {
            "model_config": model_config,
            "data": data,
            "seeds": [int(seed) for seed in seeds],
            "abba_order": list(ABBA_ORDER),
            "epochs_per_block": epochs_per_block,
            "warmup_epochs": warmup_epochs,
            "epochs_per_seed": warmup_epochs + len(ABBA_ORDER) * epochs_per_block,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": workers,
            "bootstrap": {
                "method": "percentile bootstrap of the mean (2.5%-97.5%)",
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
            },
        },
        "baselines": {
            "official_base_ref": _OFFICIAL_BASE_REF,
            "runtime_ultralytics_editable_install_head": _ULTRALYTICS_EDITABLE_HEAD,
            "baseline_root_head": _BASELINE_ROOT_HEAD,
        },
        "environment": dict(environment),
        "blocks": [dict(row) for row in blocks],
        "pairs": pairs,
        "statistics": {
            "slowdown_percent": stats,
        },
        "verdict": verdict,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default=None, help="Deployed YOLO-Master checkout (env BASELINE_ROOT or ../YOLO-Master)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model cfg path, absolute or relative to --baseline-root (default {DEFAULT_MODEL})")
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"Dataset cfg path, absolute or relative to --baseline-root (default {DEFAULT_DATA})")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds (default 0,1,2, the frozen pre-registration)")
    parser.add_argument("--epochs-per-block", type=int, default=6, help="Training epochs per ABBA block (default 6)")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Unmeasured prefix epochs per seed session (default 2)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default 640)")
    parser.add_argument("--batch", type=int, default=1, help="Training batch size (default 1, CPU)")
    parser.add_argument("--device", default="cpu", help="Training device (default cpu, the frozen environment)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (default 2)")
    parser.add_argument("--run-id", default=None, help="Artifact run_id (default timestamp-based)")
    parser.add_argument("--output", type=Path, default=None, help="Artifact JSON path (default artifacts/training_slowdown/<run-id>/slowdown_result.json)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing --output artifact")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration only; never builds a model or trains")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve and validate every configuration value shared by run modes."""
    baseline_root = Path(
        args.baseline_root
        or os.environ.get("BASELINE_ROOT")
        or str(PACKAGE_ROOT / ".." / "YOLO-Master")
    ).resolve()
    if not baseline_root.is_dir():
        raise FileNotFoundError(
            f"baseline_root not found: {baseline_root} - pass --baseline-root D:/path/to/YOLO-Master"
        )
    model = _resolve_cfg_path(baseline_root, args.model)
    data = _resolve_cfg_path(baseline_root, args.data)
    for label, path in (("model", model), ("data", data)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} cfg not found: {path} (resolved under {baseline_root})")
    seeds = _parse_seeds(args.seeds)
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"seeds must be non-negative, got {seeds}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be unique, got {seeds}")
    run_id = args.run_id or f"train-slowdown-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}"
    if args.output is None:
        output = PACKAGE_ROOT / "artifacts" / "training_slowdown" / run_id / "slowdown_result.json"
    else:
        output = Path(args.output)
    return {
        "baseline_root": baseline_root,
        "model": model,
        "data": data,
        "seeds": seeds,
        "epochs_per_block": int(args.epochs_per_block),
        "warmup_epochs": int(args.warmup_epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": str(args.device),
        "workers": int(args.workers),
        "run_id": run_id,
        "output": output,
        "layout": _block_layout(int(args.epochs_per_block), int(args.warmup_epochs)),
    }


def _print_dry_run(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    layout = cfg["layout"]
    print(f"baseline_root : {cfg['baseline_root']}")
    print(f"model cfg     : {cfg['model']}  [exists]")
    print(f"data cfg      : {cfg['data']}  [exists]")
    print(f"seeds         : {cfg['seeds']}")
    print(
        f"per-seed session: {cfg['warmup_epochs']} warmup + "
        f"{len(ABBA_ORDER)} x {cfg['epochs_per_block']} = {cfg['warmup_epochs'] + len(ABBA_ORDER) * cfg['epochs_per_block']} epochs"
    )
    print("ABBA blocks (1-based epochs, OFF/ON workload identical except observation):")
    for row in layout:
        print(
            f"  seed-block {row['block']} arm={row['arm']:>3} "
            f"epochs {row['epoch_start'] + 1}-{row['epoch_end']} "
            f"(trainer epochs {row['epoch_start']}..{row['epoch_end'] - 1})"
        )
    print("ON chain     : per train batch end of ON blocks -> MoE _moe_force_snapshot=True,"
          " routing_capture.capture_records (discover+adapt), RoutingRecordWriter JSONL append")
    print(f"output       : {cfg['output']}")
    if cfg["seeds"] != [0, 1, 2]:
        print("note         : seeds != 0,1,2 -> development check only, not the frozen pre-registered run")
    print("schema       : schema_version, run_id, started_at, finished_at, protocol, parameters, baselines,"
          " environment, blocks, pairs, statistics.slowdown_percent{mean,std,median,p95,min,max,n,ci95},"
          " verdict{ci95_upper_percent,pass}")
    print("dry-run      : OK (validation passed; no model was built, no training was run)")


def _ensure_moe_snapshot_force(root: Any) -> None:
    """Force per-forward MoE snapshot refresh (the P1-A observation convention)."""
    for _, module in root.named_modules():
        if "moe" in module.__class__.__module__.lower() and hasattr(module, "last_routing_snapshot"):
            module._moe_force_snapshot = True


def _train_seed_session(cfg: dict[str, Any], seed: int, run_id: str, output_dir: Path) -> list[dict[str, Any]]:
    """Run one real per-seed YOLO session with ABBA blocks; return block rows."""
    import torch
    from ultralytics import YOLO

    from scripts.routing_capture import capture_records
    from scripts.routing_record_writer import RoutingRecordWriter

    order = ABBA_ORDER
    epochs_per_block = cfg["epochs_per_block"]
    warmup_epochs = cfg["warmup_epochs"]
    epochs_total = warmup_epochs + len(order) * epochs_per_block

    torch.manual_seed(seed)
    model = YOLO(str(cfg["model"]))

    def _block_index(epoch: int) -> int | None:
        if epoch < warmup_epochs:
            return None
        index = (epoch - warmup_epochs) // epochs_per_block
        return index if 0 <= index < len(order) else None

    def _on_train_epoch_start(trainer: Any) -> None:
        epoch = int(trainer.epoch)
        _ensure_moe_snapshot_force(trainer.model)
        index = _block_index(epoch)
        state["epoch"] = epoch
        state["block_index"] = index
        state["arm"] = order[index] if index is not None else None
        state["epoch_started_at"] = time.perf_counter()

    def _on_train_epoch_end(trainer: Any) -> None:
        epoch = int(trainer.epoch)
        index = _block_index(epoch)
        started = state["epoch_started_at"]
        state["epoch_started_at"] = None
        if index is not None and started is not None:
            state["block_epochs"][index] += 1
            state["block_seconds"][index] += time.perf_counter() - started

    def _on_train_batch_end(trainer: Any) -> None:
        state["batch"] += 1
        index = state["block_index"]
        if index is not None:
            state["batch_calls"][index] += 1
        if index is None or state["arm"] != "ON":
            return
        records = capture_records(
            trainer.model,
            run_id=run_id,
            captured_at=_now_iso(),
            step=state["batch"],
            training=True,
        )
        for record in records:
            records_writer.write(record)
        state["records"][index] += len(records)

    state: dict[str, Any] = {
        "epoch": None,
        "block_index": None,
        "arm": None,
        "epoch_started_at": None,
        "batch": 0,
        "block_epochs": [0] * len(order),
        "block_seconds": [0.0] * len(order),
        "batch_calls": [0] * len(order),
        "records": [0] * len(order),
    }
    records_writer = RoutingRecordWriter(output_dir / f"seed{seed}_on_records.jsonl", append=True)
    model.add_callback("on_train_epoch_start", _on_train_epoch_start)
    model.add_callback("on_train_epoch_end", _on_train_epoch_end)
    model.add_callback("on_train_batch_end", _on_train_batch_end)
    try:
        model.train(
            data=str(cfg["data"]),
            epochs=epochs_total,
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=cfg["device"],
            workers=cfg["workers"],
            seed=seed,
            deterministic=True,
            patience=epochs_total + 1,
            project=str(output_dir / "runs"),
            name=f"seed{seed}",
            exist_ok=True,
            save=False,
            plots=False,
            verbose=False,
            cache=False,
        )
    finally:
        records_writer.close()

    rows = []
    for index, arm in enumerate(order):
        on_records = int(state["records"][index])
        if state["block_epochs"][index] != epochs_per_block:
            raise RuntimeError(
                f"seed {seed} block {index + 1}: expected {epochs_per_block} epochs, "
                f"got {state['block_epochs'][index]} - session did not follow the ABBA layout"
            )
        if arm == "ON" and on_records == 0:
            raise RuntimeError(
                f"seed {seed} block {index + 1} (ON): observation chain captured no routing "
                "records - the ON chain did not engage; nothing can be reported"
            )
        if arm == "OFF" and on_records != 0:
            raise RuntimeError(
                f"seed {seed} block {index + 1} (OFF): captured {on_records} records - "
                "OFF blocks must not run the observation chain"
            )
        rows.append(
            {
                "seed": seed,
                "block": index + 1,
                "arm": arm,
                "epochs": int(state["block_epochs"][index]),
                "seconds": float(state["block_seconds"][index]),
                "train_batches": int(state["batch_calls"][index]),
                "on_records": on_records,
            }
        )
    return rows


def _run_experiment(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    from scripts.measure_sample_capture_overhead import _environment

    try:
        environment = _environment()
    except Exception as exc:
        raise RuntimeError(
            "ultralytics/torch not importable - run this script with the deployed "
            f"YOLO-Master baseline Python. ({exc})"
        ) from exc
    output = cfg["output"]
    if output.exists() and not args.overwrite:
        print(f"refusing to overwrite existing output {output} (pass --overwrite)", file=sys.stderr)
        return 2
    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now_iso()
    blocks: list[dict[str, Any]] = []
    for seed in cfg["seeds"]:
        print(f"[seed {seed}] training {cfg['warmup_epochs'] + len(ABBA_ORDER) * cfg['epochs_per_block']} epochs ...", flush=True)
        rows = _train_seed_session(cfg, seed, cfg["run_id"], output_dir)
        blocks.extend(rows)
        for row in rows:
            print(
                f"  seed {seed} block {row['block']} {row['arm']}: {row['seconds']:.2f}s "
                f"({row['epochs']} epochs, {row['train_batches']} batches, {row['on_records']} on-records)"
            )
    finished_at = _now_iso()

    payload = compose_payload(
        run_id=cfg["run_id"],
        started_at=started_at,
        finished_at=finished_at,
        model_config=str(cfg["model"]),
        data=str(cfg["data"]),
        seeds=cfg["seeds"],
        epochs_per_block=cfg["epochs_per_block"],
        warmup_epochs=cfg["warmup_epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        workers=cfg["workers"],
        blocks=blocks,
        environment=environment,
    )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = payload["statistics"]["slowdown_percent"]
    verdict = payload["verdict"]
    print(
        "training slowdown: "
        f"mean {stats['mean']:.2f}% (95% CI [{stats['ci95'][0]:.2f}, {stats['ci95'][1]:.2f}], "
        f"n {stats['n']})"
    )
    print(f"verdict: {'PASS' if verdict['pass'] else 'FAIL'} (CI upper {verdict['ci95_upper_percent']:.2f}% < 10%)")
    print(f"wrote {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = resolve_config(args)
    except (OSError, ValueError) as exc:
        print(f"measure_training_slowdown: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        _print_dry_run(cfg, args)
        return 0
    try:
        return _run_experiment(cfg, args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"measure_training_slowdown: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
