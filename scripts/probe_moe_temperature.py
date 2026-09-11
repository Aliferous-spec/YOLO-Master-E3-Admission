"""MoE router temperature probe: does the public annealing API move the
discrete routing metrics?

Read-only sensitivity check for the upstream protocol-level entry point
``anneal_mixture_temperatures(model, factor=..., min_temp=...)``. It runs the
existing MoE routing case (``DetailAwareLowRankHybridAdaptiveGateMoE`` from
``yolo-master-n.yaml``) through the existing v1 collector
(``scripts.routing_capture.capture_records``) twice on one untrained model:
once as-is, once after annealing the router temperatures.

Fairness contract (temperature is the only variable between the two arms):

* one ``YOLO`` instance, one weight set, no reload and no retraining;
* the built-in cosine temperature schedule is frozen *before* the first arm,
  otherwise the upstream per-forward schedule would drift the temperature
  between arms on its own;
* BatchNorm running statistics are snapshotted once and restored before every
  arm, because train-mode forwards mutate them;
* the same input tensor and the same torch seed are used in every arm;
* every arm runs twice, so a non-zero delta cannot be attributed to RNG drift.

Observation scope: untrained (random-init) checkpoint, routing structure only.
A non-zero or zero delta here is a statement about the routing observables, not
a training-performance claim. See ``docs/moe-temperature-probe.md``.

Example:
    python -m scripts.probe_moe_temperature --out artifacts/temperature/moe-temp-factor2/moe_temperature_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

PROBE_VERSION = "e3-temperature-probe/v1"
GENERATOR = "scripts/probe_moe_temperature.py"
FIELDS = ("top1_share", "entropy_nats", "entropy_normalized", "gini")
DEFAULT_MODEL_CONFIG = "ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml"
DEFAULT_BASELINE_ROOT = "../YOLO-Master"
DEFAULT_FACTOR = 2.0
DEFAULT_MIN_TEMP = 0.3
DEFAULT_SEED = 0
DEFAULT_SIZE = 640
SCOPE = (
    "untrained random-init checkpoint; routing-structure observation only. "
    "Not a training-performance claim and not an intervention result."
)

#: v1 record attribute behind each compared field.
_RECORD_ATTR = {
    "top1_share": "dominant_expert_share",
    "entropy_nats": "routing_entropy_nats",
    "entropy_normalized": "routing_entropy_normalized",
    "gini": "load_gini",
}


# ---------------------------------------------------------------------------
# Pure helpers (no torch; covered by unit tests)
# ---------------------------------------------------------------------------


def metric_delta(baseline: Mapping[str, Any], intervention: Mapping[str, Any]) -> dict[str, float]:
    """Absolute after-minus-before delta for the recorded routing fields.

    Raises when a field is absent on either side so a changed record schema
    fails loudly instead of silently reporting a zero delta.
    """
    missing = [field for field in FIELDS if field not in baseline or field not in intervention]
    if missing:
        raise ValueError("missing routing field(s) in probe arms: " + ", ".join(sorted(missing)))
    return {field: round(float(intervention[field]) - float(baseline[field]), 6) for field in FIELDS}


def summarize_layers(
    baseline: Mapping[str, Mapping[str, Any]], intervention: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per-layer before/after plus delta; fails loudly on a missing layer."""
    layers: dict[str, dict[str, Any]] = {}
    for name in sorted(baseline):
        if name not in intervention:
            raise ValueError(f"intervention arm is missing module {name!r}")
        before = baseline[name]
        after = intervention[name]
        layers[name] = {
            "module_type": before.get("module_type"),
            "num_experts": before.get("num_experts"),
            "top_k": before.get("top_k"),
            "baseline": {field: before[field] for field in FIELDS},
            "factor_2_0": {field: after[field] for field in FIELDS},
            "delta": metric_delta(before, after),
            "dominant_expert": {
                "baseline": before.get("dominant_expert"),
                "factor_2_0": after.get("dominant_expert"),
            },
            "dominant_expert_changed": before.get("dominant_expert") != after.get("dominant_expert"),
            "expert_usage": {"baseline": before.get("usage"), "factor_2_0": after.get("usage")},
        }
    return layers


def aggregate_deltas(layers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Mean delta and changed-layer count per field, across all MoE layers."""
    deltas = [layer["delta"] for layer in layers.values()]
    count = len(deltas)
    return {
        "moe_layers": count,
        "changed_layers": {field: sum(1 for delta in deltas if delta[field] != 0.0) for field in FIELDS},
        "dominant_expert_changed_layers": sum(1 for layer in layers.values() if layer["dominant_expert_changed"]),
        "mean_delta": {
            field: round(sum(delta[field] for delta in deltas) / count, 6) if count else 0.0 for field in FIELDS
        },
    }


def arm_digest(rows: Mapping[str, Mapping[str, Any]]) -> str:
    """SHA-256 of one arm's layer rows, so a re-run can be compared exactly."""
    canonical = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Model-facing helpers
# ---------------------------------------------------------------------------


def temperature_holders(model: Any) -> list[tuple[str, Any]]:
    """Modules the anneal API targets: mixture package + a temperature attribute."""
    return [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if "moe" in module.__class__.__module__.lower() and hasattr(module, "temperature")
    ]


def router_temperatures(model: Any) -> dict[str, float]:
    """Current temperature per holder, for both float and tensor representations."""
    out = {}
    for name, module in temperature_holders(model):
        value = module.temperature
        out[name] = round(float(value.detach()) if hasattr(value, "detach") else float(value), 6)
    return out


def weights_digest(model: Any) -> str:
    """SHA-256 over the full state dict, taken before any arm runs.

    Train-mode forwards mutate BatchNorm buffers, so the digest is only
    meaningful when it is computed on the freshly built model; it is the
    evidence that every arm ran on one identical weight set.
    """
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def moe_metric_rows(records: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Reduce MoE v1 records to the compared discrete-routing fields."""
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.family != "moe":
            continue
        routing = record.routing
        rows[record.module.name] = {
            "module_type": record.module.type,
            "num_experts": int(routing.num_experts),
            "top_k": int(routing.top_k),
            "dominant_expert": int(routing.dominant_expert),
            "usage": [round(float(value), 6) for value in routing.expert_usage],
            **{field: round(float(getattr(routing, _RECORD_ATTR[field])), 6) for field in FIELDS},
        }
    return rows


def run_probe(
    *,
    baseline_root: str,
    model_config: str,
    factor: float,
    min_temp: float,
    seed: int,
    size: int,
) -> dict[str, Any]:
    """Run the four arms (two per temperature) and return the probe payload."""
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules.routing_protocol import (
        anneal_mixture_temperatures,
        configure_mixture_temperature_schedule,
    )

    from scripts.routing_capture import (
        capture_records,
        restore_bn_running_state,
        snapshot_bn_running_state,
    )
    from scripts.run_e3_smoke import collect_environment

    resolved = str(Path(ultralytics.__file__).resolve())
    root = str(Path(baseline_root).resolve())
    if not resolved.lower().startswith(root.lower()):
        raise RuntimeError(
            f"import gate failed: ultralytics resolved to {resolved}, expected a checkout under {root}. "
            "Put the baseline checkout ahead of any editable install (e.g. PYTHONPATH=D:/YOLO-Master)."
        )

    config_path = Path(baseline_root) / model_config
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")

    torch.manual_seed(seed)
    model = YOLO(str(config_path))
    module = model.model

    forced = 0
    for _, candidate in module.named_modules():
        if "moe" in candidate.__class__.__module__.lower() and hasattr(candidate, "last_routing_snapshot"):
            candidate._moe_force_snapshot = True
            forced += 1
    if forced == 0:
        raise RuntimeError("no routed MoE modules to force snapshots on")

    frozen = configure_mixture_temperature_schedule(module, external=True)
    module.train()
    checkpoint_digest = weights_digest(module)
    bn_state = snapshot_bn_running_state(module)
    if not bn_state:
        raise RuntimeError("no track_running_stats BatchNorm found to isolate")

    torch.manual_seed(seed)
    tensor = torch.randn(1, 3, int(size), int(size))

    def arm(tag: str) -> dict[str, dict[str, Any]]:
        restore_bn_running_state(module, bn_state)
        torch.manual_seed(seed)
        with torch.no_grad():
            module(tensor)
        return moe_metric_rows(capture_records(module, run_id=tag, captured_at="probe", training=True))

    temperatures_before = router_temperatures(module)
    baseline = arm("probe-baseline")
    baseline_repeat = arm("probe-baseline-repeat")

    updated = anneal_mixture_temperatures(module, factor=factor, min_temp=min_temp)
    temperatures_after = router_temperatures(module)
    intervention = arm("probe-factor-2.0")
    intervention_repeat = arm("probe-factor-2.0-repeat")

    layers = summarize_layers(baseline, intervention)
    return {
        "schema_version": PROBE_VERSION,
        "generator": GENERATOR,
        "purpose": "routing-only sensitivity probe for anneal_mixture_temperatures",
        "scope": SCOPE,
        "checkpoint": {
            "source": "random_init_from_yaml",
            "weights_sha256": checkpoint_digest,
            "shared_by_all_arms": True,
        },
        "environment": collect_environment(),
        "config": {
            "baseline_root": str(Path(baseline_root).resolve()),
            "model_config": str(config_path),
            "factor": factor,
            "min_temp": min_temp,
            "seed": seed,
            "image_size": int(size),
            "device": "cpu",
            "input": "torch.randn(1, 3, size, size), one tensor reused by every arm",
            "observer": "scripts.routing_capture.capture_records (e3-routing/v1, family=moe)",
            "temperature_schedule_frozen_modules": frozen,
            "forced_snapshot_modules": forced,
        },
        "anneal": {
            "updated_modules": updated,
            "temperature_before": temperatures_before,
            "temperature_after": temperatures_after,
        },
        "layers": layers,
        "aggregate": aggregate_deltas(layers),
        "determinism": {
            "baseline_repeat_identical": baseline == baseline_repeat,
            "factor_2_0_repeat_identical": intervention == intervention_repeat,
            "arm_sha256": {
                "baseline": arm_digest(baseline),
                "baseline_repeat": arm_digest(baseline_repeat),
                "factor_2_0": arm_digest(intervention),
                "factor_2_0_repeat": arm_digest(intervention_repeat),
            },
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-root", default=DEFAULT_BASELINE_ROOT, help="YOLO-Master checkout to probe")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="model YAML relative to the baseline root")
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR, help="anneal factor passed to the API")
    parser.add_argument("--min-temp", type=float, default=DEFAULT_MIN_TEMP, help="min_temp passed to the API")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--out", required=True, help="path of the JSON result file to write")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_probe(
        baseline_root=args.baseline_root,
        model_config=args.model_config,
        factor=args.factor,
        min_temp=args.min_temp,
        seed=args.seed,
        size=args.size,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "anneal": payload["anneal"], "aggregate": payload["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())