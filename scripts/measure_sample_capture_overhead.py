#!/usr/bin/env python3
"""Measure the P1-A per-sample capture chain overhead and write the P1-B artifact.

Structure mirrors ``measure_routing_hook_overhead.py``; the measured object is
the P1-A per-sample capture path on the MoE model (the chain that also carries
BatchNorm running-state restore and forced snapshot refresh):

- OFF arm: plain train-mode model forward (``iterations`` forwards).
- ON arm: ``iterations`` P1-A sample capture cycles, each = BN running-state
  restore + forced MoE snapshot refresh + forward + module discovery/adapter
  -> in-memory v1 records.  JSONL serialization is intentionally excluded
  (spec section 7 keeps it out of the measured arms).

Each independent ON/OFF pair is repeated ``--repeats`` times; the artifact
records protocol metadata, the baseline triple, environment, and per-pair
overhead statistics (mean +/- std, min/max, n).

Example:
    python scripts/measure_sample_capture_overhead.py --run-id smoke-x --output sample_overhead_result.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Baseline triple recorded by P0/P1 evidence (docs README "版本与边界" / p1-spec
# section 7): official locked config ref, runtime ultralytics editable install
# HEAD, and the smoke baseline_root checkout HEAD.
_OFFICIAL_BASE_REF = "3eb6cd914b651a06e2cd08ea87d12c28cab95502"
_ULTRALYTICS_EDITABLE_HEAD = "d604c4b"
_BASELINE_ROOT_HEAD = "aa5d2e2"
# P1-A per-family sample workload (spec section 1 conservative scope).
_SAMPLE_WORKLOAD = {"mot": 4, "moe": 4, "latent": 1}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _summarize(values: Sequence[float]) -> dict[str, float]:
    """Return mean / std / min / max / n over a finite sample (n >= 2)."""
    finite = [float(value) for value in values]
    if len(finite) < 2:
        raise ValueError("need at least 2 samples to summarize")
    if not all(math.isfinite(value) for value in finite):
        raise ValueError("cannot summarize non-finite values")
    return {
        "mean": float(statistics.fmean(finite)),
        "std": float(statistics.stdev(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
        "n": len(finite),
    }


def _environment() -> dict[str, str]:
    import platform

    import torch
    import ultralytics

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }


def _repeat_payload(index: int, off_seconds: float, on_seconds: float, iterations: int) -> dict[str, float]:
    if not (math.isfinite(off_seconds) and math.isfinite(on_seconds)) or off_seconds <= 0.0:
        raise ValueError("arm timings must be finite and off_seconds > 0")
    overhead = (on_seconds - off_seconds) / off_seconds * 100.0
    if not math.isfinite(overhead):
        raise ValueError("overhead_percent must be finite")
    return {
        "repeat": index + 1,
        "off_seconds": float(off_seconds),
        "on_seconds": float(on_seconds),
        "overhead_percent": float(overhead),
        "off_ms_per_iteration": float(off_seconds / iterations * 1000.0),
        "on_ms_per_iteration": float(on_seconds / iterations * 1000.0),
    }


def compose_payload(
    *,
    run_id: str,
    captured_at: str,
    started_at: str,
    finished_at: str,
    model_config: str,
    iterations: int,
    warmup: int,
    repeats: int,
    size: int,
    arm_pairs: Sequence[tuple[float, float]],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the P1-B artifact payload from one measured run."""
    if len(arm_pairs) != repeats:
        raise ValueError(f"arm_pairs {len(arm_pairs)} != repeats {repeats}")
    repetitions = [
        _repeat_payload(index, off_seconds, on_seconds, iterations)
        for index, (off_seconds, on_seconds) in enumerate(arm_pairs)
    ]
    return {
        "run_id": run_id,
        "captured_at": captured_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "protocol": {
            "name": "p1-b-sample-overhead",
            "measured_object": (
                "P1-A per-sample capture chain on the MoE yolo-master-n model "
                "(snapshot force + BatchNorm running-state restore + forward + "
                "discovery/adapter -> in-memory v1 records); MoT/Latent sample "
                "capture reuse the same primitives on plain eval forwards"
            ),
            "arms": {
                "off": "plain train-mode model forward",
                "on": "P1-A per-sample capture cycle (BN restore + snapshot force + "
                "forward + adapters; JSONL write excluded)",
            },
            "unit": {
                "overhead": "percent",
                "latency": "seconds",
                "per_iteration": "milliseconds per forward / capture cycle",
            },
            "sample_input": {
                "shape": [1, 3, size, size],
                "source": "random tensor (same shape as the P1-A MoE coco8 val letterboxed input)",
                "mode": "train",
            },
            "sample_workload": dict(_SAMPLE_WORKLOAD),
            "device": "cpu",
            "threshold": "none (statistics only; no pre-registered <10% judgment)",
        },
        "parameters": {
            "model_config": model_config,
            "iterations_per_arm": iterations,
            "warmup_iterations": warmup,
            "independent_repeats": repeats,
            "size": size,
        },
        "baselines": {
            "official_base_ref": _OFFICIAL_BASE_REF,
            "runtime_ultralytics_editable_install_head": _ULTRALYTICS_EDITABLE_HEAD,
            "baseline_root_head": _BASELINE_ROOT_HEAD,
        },
        "environment": dict(environment),
        "statistics": {
            "overhead_percent": _summarize([rep["overhead_percent"] for rep in repetitions]),
            "off_seconds": _summarize([rep["off_seconds"] for rep in repetitions]),
            "on_seconds": _summarize([rep["on_seconds"] for rep in repetitions]),
        },
        "repetitions": repetitions,
    }


def _timed(fn: Any, iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    return time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml",
        help="MoE model YAML to measure (same default as the P0 overhead step)",
    )
    parser.add_argument("--iterations", type=int, default=50, help="forwards / capture cycles per arm")
    parser.add_argument("--warmup", type=int, default=5, help="warmup forwards before timing")
    parser.add_argument("--repeats", type=int, default=3, help="independent on/off pairs")
    parser.add_argument("--size", type=int, default=640, help="input resolution (square)")
    parser.add_argument("--run-id", required=True, help="smoke run_id recorded in the artifact")
    parser.add_argument("--output", required=True, type=Path, help="path to write sample_overhead_result.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from ultralytics import YOLO

    from scripts.routing_capture import (
        capture_sample_records,
        restore_bn_running_state,
        snapshot_bn_running_state,
    )

    started_at = _now_iso()
    captured_at = started_at
    model = YOLO(args.model)
    module = model.model.train()

    forced = 0
    for _, mod in module.named_modules():
        if "moe" in mod.__class__.__module__.lower() and hasattr(mod, "last_routing_snapshot"):
            mod._moe_force_snapshot = True
            forced += 1
    if forced == 0:
        raise SystemExit("measure_sample_capture_overhead: no routed MoE modules to force snapshots")
    bn_snapshot = snapshot_bn_running_state(module)
    if not bn_snapshot:
        raise SystemExit("measure_sample_capture_overhead: no track_running_stats BatchNorm to isolate")

    x = torch.randn(1, 3, int(args.size), int(args.size))
    held_records: list[Any] = []

    def off_iteration() -> None:
        module(x)

    def on_iteration() -> None:
        held_records.extend(
            capture_sample_records(
                module,
                [x],
                run_id=args.run_id,
                captured_at=captured_at,
                training=True,
                before_each=lambda root, _step: restore_bn_running_state(root, bn_snapshot),
            )
        )

    for _ in range(args.warmup):
        off_iteration()
    for _ in range(args.warmup):
        on_iteration()

    arm_pairs: list[tuple[float, float]] = []
    for _ in range(args.repeats):
        off_seconds = _timed(off_iteration, args.iterations)
        on_seconds = _timed(on_iteration, args.iterations)
        arm_pairs.append((off_seconds, on_seconds))
        print(f"rep {len(arm_pairs)}: off={off_seconds:.3f}s on={on_seconds:.3f}s")

    try:
        payload = compose_payload(
            run_id=args.run_id,
            captured_at=captured_at,
            started_at=started_at,
            finished_at=_now_iso(),
            model_config=args.model,
            iterations=args.iterations,
            warmup=args.warmup,
            repeats=args.repeats,
            size=int(args.size),
            arm_pairs=arm_pairs,
            environment=_environment(),
        )
    except ValueError as exc:
        raise SystemExit(f"measure_sample_capture_overhead: {exc}") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = payload["statistics"]["overhead_percent"]
    print(
        "sample overhead: "
        f"mean {stats['mean']:.2f}% +/- {stats['std']:.2f}% "
        f"(min {stats['min']:.2f}, max {stats['max']:.2f}, n {stats['n']})"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())