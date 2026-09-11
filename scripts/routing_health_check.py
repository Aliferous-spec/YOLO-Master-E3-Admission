"""Routing health check: turn the known expert-collapse phenomenon into a
reproducible, read-only diagnostic.

Reads an already-finished run's ``routing_records.jsonl`` (canonical) -- or any
``e3-routing/v1`` JSONL -- and reports, per module: Gini, normalized entropy,
top-1 expert share, the dominant expert, and an explicit collapse verdict. It
never trains, never touches a model or a forward pass, and never writes back
into the source run directory.

Explicit thresholds (also in ``docs/routing-health-check.md``)
    COLLAPSE_SHARE_THRESHOLD = 0.80  top-1 share >= this           => collapse
    ONE_HOT_SHARE_THRESHOLD  = 0.999 ... and entropy_norm <= ENTROPY_COLLAPSE_MAX
    ENTROPY_COLLAPSE_MAX     = 0.05                                  => "one_hot"
    UNIFORM_ENTROPY_MIN      = 0.99  entropy_norm >= this and gini <= UNIFORM_GINI_MAX
    UNIFORM_GINI_MAX         = 0.01                                  => "uniform"
The 0.80 share threshold is the value already suggested for the schema's
``collapse_flag`` in ``docs/smoke-design-and-schema.md``; the remaining
constants only name the one-hot / uniform anchors recorded in
``docs/p0-three-seed-evidence.md``.

Diagnostic only: this reports a structural property of a routing snapshot. It is
not a performance claim and does not change any experiment conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.routing_panel_sink import load_records
from scripts.routing_record import RoutingRecord

logger = logging.getLogger("e3-routing-health")

CHECK_VERSION = "1.0.0"
GENERATOR = "scripts/routing_health_check.py"
DEFAULT_RECORDS_NAME = "routing_records.jsonl"

COLLAPSE_SHARE_THRESHOLD = 0.80
ONE_HOT_SHARE_THRESHOLD = 0.999
ENTROPY_COLLAPSE_MAX = 0.05
UNIFORM_ENTROPY_MIN = 0.99
UNIFORM_GINI_MAX = 0.01
GINI_TOLERANCE = 1e-6


def thresholds() -> dict[str, float]:
    """The exact thresholds the verdicts below are computed from."""
    return {
        "collapse_share_threshold": COLLAPSE_SHARE_THRESHOLD,
        "one_hot_share_threshold": ONE_HOT_SHARE_THRESHOLD,
        "entropy_collapse_max": ENTROPY_COLLAPSE_MAX,
        "uniform_entropy_min": UNIFORM_ENTROPY_MIN,
        "uniform_gini_max": UNIFORM_GINI_MAX,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_sort_key(name: str) -> tuple[int, str]:
    """Sort ``model.<n>[.routing]`` module names by their numeric index."""
    tail = name.split(".")[1] if "." in name else name
    try:
        return (int(tail), name)
    except ValueError:
        return (1 << 30, name)


def _resolve_run_id(run_dir: Path) -> str:
    summary = Path(run_dir) / "summary.json"
    if summary.is_file():
        try:
            run_id = json.loads(summary.read_text(encoding="utf-8")).get("run_id")
            if run_id:
                return str(run_id)
        except (json.JSONDecodeError, OSError):
            pass
    return Path(run_dir).name


def classify_module(
    *, num_experts: int, gini: float, entropy_normalized: float, top1_share: float
) -> dict[str, Any]:
    """Apply the documented thresholds to one module's routing snapshot."""
    one_hot_reference = (num_experts - 1) / num_experts if num_experts > 1 else 0.0
    collapsed = top1_share >= COLLAPSE_SHARE_THRESHOLD
    one_hot = top1_share >= ONE_HOT_SHARE_THRESHOLD and entropy_normalized <= ENTROPY_COLLAPSE_MAX
    uniform = entropy_normalized >= UNIFORM_ENTROPY_MIN and gini <= UNIFORM_GINI_MAX
    if one_hot:
        severity = "one_hot"
    elif collapsed:
        severity = "concentrated"
    else:
        severity = "spread"
    return {
        "collapse": bool(collapsed),
        "severity": severity,
        "one_hot": bool(one_hot),
        "uniform": bool(uniform),
        "gini_one_hot_reference": one_hot_reference,
        "gini_matches_one_hot": abs(gini - one_hot_reference) <= GINI_TOLERANCE,
    }


def diagnose_records(
    records: Sequence[RoutingRecord], *, run_id: str = "", source: str = ""
) -> dict[str, Any]:
    """Diagnose an in-memory record list; one module row per (family, layer)."""
    if not records:
        raise ValueError(f"no routing records to diagnose ({source or 'in-memory input'})")
    grouped: dict[tuple[str, str], list[RoutingRecord]] = {}
    for record in records:
        grouped.setdefault((record.family, record.module.name), []).append(record)

    modules: list[dict[str, Any]] = []
    for (family, layer), group in grouped.items():
        count = len(group)
        worst = max(group, key=lambda item: item.routing.dominant_expert_share)
        gini = sum(item.routing.load_gini for item in group) / count
        entropy_normalized = sum(item.routing.routing_entropy_normalized for item in group) / count
        entropy_nats = sum(item.routing.routing_entropy_nats for item in group) / count
        top1_share = worst.routing.dominant_expert_share
        modules.append(
            {
                "family": family,
                "layer": layer,
                "module_type": worst.module.type,
                "routing_paradigm": worst.routing_paradigm,
                "num_experts": worst.routing.num_experts,
                "top_k": worst.routing.top_k,
                "gini": gini,
                "entropy_normalized": entropy_normalized,
                "entropy_nats": entropy_nats,
                "top1_share": top1_share,
                "dominant_expert": worst.routing.dominant_expert,
                "records": count,
                **classify_module(
                    num_experts=worst.routing.num_experts,
                    gini=gini,
                    entropy_normalized=entropy_normalized,
                    top1_share=top1_share,
                ),
            }
        )
    modules.sort(key=lambda module: (module["family"], _module_sort_key(module["layer"])))

    by_family: dict[str, dict[str, int]] = {}
    for module in modules:
        bucket = by_family.setdefault(module["family"], {"modules": 0, "collapsed": 0})
        bucket["modules"] += 1
        bucket["collapsed"] += int(module["collapse"])
    summary = {
        "modules": len(modules),
        "collapsed": sum(int(module["collapse"]) for module in modules),
        "one_hot": sum(int(module["one_hot"]) for module in modules),
        "uniform": sum(int(module["uniform"]) for module in modules),
        "by_family": by_family,
    }
    source_path = Path(source) if source else None
    return {
        "check_version": CHECK_VERSION,
        "generator": GENERATOR,
        "run_id": run_id,
        "source": source,
        "source_sha256": _sha256(source_path) if source_path and source_path.is_file() else None,
        "thresholds": thresholds(),
        "summary": summary,
        "modules": modules,
    }


def check_run(run_dir: Path, *, records_name: str = DEFAULT_RECORDS_NAME) -> dict[str, Any]:
    """Diagnose one finished run directory (read-only)."""
    run_dir = Path(run_dir)
    records_path = run_dir / records_name
    if not records_path.is_file():
        raise FileNotFoundError(f"missing records file: {records_path}")
    return diagnose_records(
        load_records(records_path),
        run_id=_resolve_run_id(run_dir),
        source=records_path.as_posix(),
    )


def render_text_summary(result: dict[str, Any]) -> str:
    """Human-readable one-block summary (the JSON stays the machine surface)."""
    summary = result["summary"]
    limits = result["thresholds"]
    threshold_line = (
        f"thresholds: collapse top1_share >= {limits['collapse_share_threshold']:.2f}; "
        f"one_hot >= {limits['one_hot_share_threshold']:.3f} and entropy_norm <= "
        f"{limits['entropy_collapse_max']:.2f}; uniform entropy_norm >= "
        f"{limits['uniform_entropy_min']:.2f} and gini <= {limits['uniform_gini_max']:.2f}"
    )
    summary_line = (
        f"modules: {summary['modules']} | collapsed {summary['collapsed']} | "
        f"one_hot {summary['one_hot']} | uniform {summary['uniform']}"
    )
    lines = [
        f"routing health check - run {result['run_id'] or '-'}",
        f"source: {result['source'] or '-'}",
        threshold_line,
        summary_line,
    ]
    for module in result["modules"]:
        tag = " (collapse)" if module["collapse"] else " (uniform)" if module["uniform"] else ""
        lines.append(
            f"  [{module['family']:<6}] {module['layer']:<16} E={module['num_experts']:<3} "
            f"top1={module['top1_share']:.3f} dom={module['dominant_expert']:<3} "
            f"gini={module['gini']:.4f} H_norm={module['entropy_normalized']:.4f} "
            f"-> {module['severity']}{tag}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only routing health check for one E3 run.")
    parser.add_argument("--run-dir", required=True, help="artifacts/smoke/<run_id> directory")
    parser.add_argument("--records", default=DEFAULT_RECORDS_NAME, help="records filename inside the run dir")
    parser.add_argument("--out", default="", help="write the JSON diagnostic here (default: stdout only)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of the text summary")
    args = parser.parse_args(argv)

    result = check_run(Path(args.run_dir), records_name=args.records)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        logger.info("wrote diagnostic: %s", out_path)
    print(rendered if args.json else render_text_summary(result))
    return 0


__all__ = [
    "CHECK_VERSION",
    "COLLAPSE_SHARE_THRESHOLD",
    "DEFAULT_RECORDS_NAME",
    "GENERATOR",
    "ONE_HOT_SHARE_THRESHOLD",
    "check_run",
    "classify_module",
    "diagnose_records",
    "render_text_summary",
    "thresholds",
]


if __name__ == "__main__":
    raise SystemExit(main())