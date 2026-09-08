#!/usr/bin/env python3
"""Measure the P1-A per-sample capture chain overhead and write the P1-B artifact.

Structure mirrors ``measure_routing_hook_overhead.py``; the measured object is
the P1-A per-sample capture path on the MoE model (the chain that also carries
BatchNorm running-state restore and forced snapshot refresh):

- OFF arm: one plain train-mode model forward.
- ON arm: one P1-A sample capture cycle = BN running-state restore + forced
  MoE snapshot refresh + forward + module discovery/adapter -> in-memory v1
  records.  JSONL serialization is intentionally excluded (spec section 7
  keeps it out of the measured arms).

Protocol: iteration-level alternating paired observations.  Each observation
times one OFF iteration followed immediately by one ON iteration, so slow
machine drift stays inside a single observation instead of being subtracted
across minutes (the superseded design timed whole 50-iteration OFF then ON
blocks and left drift in the difference).  The spec (docs/p1-spec.md section
7) only requires >= 3 independent on/off repeats and defines no ABBA or
order-balancing parameter, so a fixed OFF->ON order is used with that stated
limitation.

Each run records ``--pairs`` paired observations (default 120, target >= 100).
Artifact statistics: per-observation ``overhead_percent`` (paired difference
normalized by the OFF arm), ``paired_difference_ms`` (ON - OFF), and the
OFF/ON ms per iteration.  Reported as mean +/- std, median, P95, min/max, n,
plus a percentile bootstrap 95% CI of the mean overhead (deterministic seed).
No <10% (or any) performance conclusion is drawn: the artifact only reports
measured statistics.

Example:
    python scripts/measure_sample_capture_overhead.py --run-id smoke-x --output sample_overhead_result.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
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

# Baseline triple recorded by P0/P1 evidence (docs README "?????" / p1-spec
# section 7): official locked config ref, runtime ultralytics editable install
# HEAD, and the smoke baseline_root checkout HEAD.
_OFFICIAL_BASE_REF = "3eb6cd914b651a06e2cd08ea87d12c28cab95502"
_ULTRALYTICS_EDITABLE_HEAD = "d604c4b"
_BASELINE_ROOT_HEAD = "aa5d2e2"
# P1-A per-family sample workload (spec section 1 conservative scope).
_SAMPLE_WORKLOAD = {"mot": 4, "moe": 4, "latent": 1}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _percentile_nearest_rank(sorted_values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile over an ascending sequence (0 < p <= 100)."""
    if not sorted_values or not 0.0 < percentile <= 100.0:
        raise ValueError("percentile requires values and 0 < percentile <= 100")
    index = max(0, math.ceil(percentile / 100.0 * len(sorted_values)) - 1)
    return float(sorted_values[index])


def _distribution(values: Sequence[float]) -> dict[str, float]:
    """Return mean / std / median / p95 / min / max / n over finite values."""
    finite = [float(value) for value in values]
    if len(finite) < 2:
        raise ValueError("need at least 2 samples to summarize")
    if not all(math.isfinite(value) for value in finite):
        raise ValueError("cannot summarize non-finite values")
    ordered = sorted(finite)
    return {
        "mean": float(statistics.fmean(finite)),
        "std": float(statistics.stdev(finite)),
        "median": float(statistics.median(finite)),
        "p95": _percentile_nearest_rank(ordered, 95.0),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "n": len(finite),
    }


def _bootstrap_ci_mean(values: Sequence[float], *, resamples: int, seed: int) -> list[float]:
    """Percentile bootstrap 95% CI of the mean over ``values`` (deterministic)."""
    finite = [float(value) for value in values]
    if len(finite) < 2:
        raise ValueError("need at least 2 samples to bootstrap")
    if not all(math.isfinite(value) for value in finite):
        raise ValueError("cannot bootstrap non-finite values")
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(finite, k=len(finite))) for _ in range(resamples))
    return [float(means[int(resamples * 0.025)]), float(means[int(resamples * 0.975) - 1])]


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


def _observation_payload(index: int, off_seconds: float, on_seconds: float) -> dict[str, float]:
    """One paired OFF/ON observation: raw times, overhead %, and ms difference."""
    if not (math.isfinite(off_seconds) and math.isfinite(on_seconds)) or off_seconds <= 0.0:
        raise ValueError("arm timings must be finite and off_seconds > 0")
    overhead = (on_seconds - off_seconds) / off_seconds * 100.0
    if not math.isfinite(overhead):
        raise ValueError("overhead_percent must be finite")
    return {
        "pair": index + 1,
        "off_seconds": float(off_seconds),
        "on_seconds": float(on_seconds),
        "overhead_percent": float(overhead),
        "off_ms_per_iteration": float(off_seconds * 1000.0),
        "on_ms_per_iteration": float(on_seconds * 1000.0),
        "difference_ms": float((on_seconds - off_seconds) * 1000.0),
    }


def compose_payload(
    *,
    run_id: str,
    captured_at: str,
    started_at: str,
    finished_at: str,
    model_config: str,
    pairs: int,
    warmup: int,
    size: int,
    arm_pairs: Sequence[tuple[float, float]],
    environment: Mapping[str, str],
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Assemble the P1-B artifact payload from one measured run."""
    if len(arm_pairs) != pairs:
        raise ValueError(f"arm_pairs {len(arm_pairs)} != pairs {pairs}")
    observations = [
        _observation_payload(index, off_seconds, on_seconds)
        for index, (off_seconds, on_seconds) in enumerate(arm_pairs)
    ]
    overhead_values = [obs["overhead_percent"] for obs in observations]
    overhead_stats = _distribution(overhead_values)
    overhead_stats["ci95"] = _bootstrap_ci_mean(
        overhead_values, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    return {
        "run_id": run_id,
        "captured_at": captured_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "protocol": {
            "name": "p1-b-sample-overhead",
            "design": "iteration-level alternating paired OFF/ON observations",
            "pairing": {
                "order": "OFF then ON within each observation",
                "rationale": (
                    "adjacent pairing bounds slow machine drift within one "
                    "observation; spec section 7 defines no ABBA parameter, "
                    "so a single OFF->ON order is used"
                ),
            },
            "measured_object": (
                "P1-A per-sample capture chain on the MoE yolo-master-n model "
                "(snapshot force + BatchNorm running-state restore + forward + "
                "discovery/adapter -> in-memory v1 records); MoT/Latent sample "
                "capture reuse the same primitives on plain eval forwards"
            ),
            "arms": {
                "off": "one plain train-mode model forward",
                "on": "one P1-A sample capture cycle (BN restore + snapshot force + "
                "forward + adapters; JSONL write excluded)",
            },
            "unit": {
                "overhead": "percent of off-arm time (paired difference)",
                "latency": "seconds",
                "per_iteration": "milliseconds per forward / capture cycle",
                "difference": "milliseconds (on - off) within an observation",
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
            "warmup_iterations": warmup,
            "paired_observations": pairs,
            "workload_per_observation": "1 OFF forward + 1 ON P1-A capture cycle",
            "size": size,
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
        "statistics": {
            "overhead_percent": overhead_stats,
            "paired_difference_ms": _distribution([obs["difference_ms"] for obs in observations]),
            "off_ms_per_iteration": _distribution([obs["off_ms_per_iteration"] for obs in observations]),
            "on_ms_per_iteration": _distribution([obs["on_ms_per_iteration"] for obs in observations]),
        },
        "observations": observations,
    }


def _time_once(fn: Any) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml",
        help="MoE model YAML to measure (same default as the P0 overhead step)",
    )
    parser.add_argument(
        "--pairs", type=int, default=120, help="paired OFF/ON observations to record (>= 100)"
    )
    parser.add_argument("--warmup", type=int, default=5, help="warmup forwards per arm before timing")
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
    for pair in range(args.pairs):
        off_seconds = _time_once(off_iteration)
        on_seconds = _time_once(on_iteration)
        arm_pairs.append((off_seconds, on_seconds))
        if (pair + 1) % 25 == 0 or pair + 1 == args.pairs:
            print(f"pair {pair + 1}/{args.pairs}: off={off_seconds * 1000:.1f}ms on={on_seconds * 1000:.1f}ms")

    try:
        payload = compose_payload(
            run_id=args.run_id,
            captured_at=captured_at,
            started_at=started_at,
            finished_at=_now_iso(),
            model_config=args.model,
            pairs=args.pairs,
            warmup=args.warmup,
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
    difference = payload["statistics"]["paired_difference_ms"]
    print(
        "sample overhead: "
        f"mean {stats['mean']:.2f}% (95% CI [{stats['ci95'][0]:.2f}, {stats['ci95'][1]:.2f}]) "
        f"median {stats['median']:.2f}% p95 {stats['p95']:.2f}% "
        f"(min {stats['min']:.2f}, max {stats['max']:.2f}, n {stats['n']})"
    )
    print(
        "paired difference ms: "
        f"mean {difference['mean']:.2f} (median {difference['median']:.2f}, p95 {difference['p95']:.2f})"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
