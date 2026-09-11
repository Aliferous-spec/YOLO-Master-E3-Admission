"""Tests for the read-only routing health check.

The check is a diagnostic surface: it must report the metrics it read (never
invent them), flag the documented collapse thresholds, and fail loudly on empty
or missing input instead of implying a healthy run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.routing_health_check import (
    COLLAPSE_SHARE_THRESHOLD,
    check_run,
    classify_module,
    diagnose_records,
)
from scripts.routing_record import RoutingRecord


def _metrics(usage: list[float]) -> tuple[float, float, float, int, float]:
    total = sum(usage)
    shares = [value / total for value in usage]
    nats = -sum(share * math.log(share) for share in shares if share > 0)
    count = len(usage)
    normalized = nats / math.log(count) if count > 1 else 0.0
    ordered = sorted(usage)
    gini = 2 * sum((index + 1) * value for index, value in enumerate(ordered)) / (count * total) - (count + 1) / count
    gini = min(max(gini, 0.0), 1.0)
    dominant = max(range(count), key=lambda index: shares[index])
    return nats, normalized, gini, dominant, shares[dominant]


def _line(
    family: str,
    module: str,
    usage: list[float],
    *,
    module_type: str,
    paradigm: str,
    run_id: str = "health-run",
) -> str:
    nats, normalized, gini, dominant, share = _metrics(usage)
    return RoutingRecord.from_dict(
        {
            "schema_version": "e3-routing/v1",
            "run_id": run_id,
            "captured_at": "2026-09-05T20:45:55+08:00",
            "family": family,
            "routing_paradigm": paradigm,
            "module": {"name": module, "type": module_type},
            "routing": {
                "num_experts": len(usage),
                "top_k": 1,
                "expert_usage": list(usage),
                "normalized_expert_load": [value / sum(usage) for value in usage],
                "routing_entropy_nats": nats,
                "routing_entropy_normalized": normalized,
                "load_gini": gini,
                "dominant_expert": dominant,
                "dominant_expert_share": share,
            },
        }
    ).to_json()


def _write_run(root: Path, run_id: str, lines: list[str]) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (run_dir / "routing_records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def test_check_run_reports_metrics_and_traceable_source(tmp_path: Path) -> None:
    lines = [
        _line("moe", "model.5.routing", [1.0, 1.0, 0.0, 0.0], module_type="MoE", paradigm="discrete_selection"),
        _line("latent", "model.23", [1.0, 1.0, 1.0, 1.0], module_type="LatentMixture", paradigm="continuous_fusion"),
    ]
    run_dir = _write_run(tmp_path, "health-fixture", lines)

    result = check_run(run_dir)

    assert result["run_id"] == "health-fixture"
    assert result["source_sha256"] == hashlib.sha256((run_dir / "routing_records.jsonl").read_bytes()).hexdigest()
    moe = next(module for module in result["modules"] if module["family"] == "moe")
    assert moe["collapse"] is False
    assert moe["severity"] == "spread"
    assert moe["top1_share"] == pytest.approx(0.5)
    latent = next(module for module in result["modules"] if module["family"] == "latent")
    assert latent["uniform"] is True
    assert latent["collapse"] is False
    assert result["summary"]["collapsed"] == 0


def test_diagnose_flags_one_hot_collapse() -> None:
    usage = [0.0] * 16
    usage[0] = 1.0
    records = [
        RoutingRecord.from_json(_line("moe", "model.11.routing", usage, module_type="MoE", paradigm="discrete_selection"))
    ]

    module = diagnose_records(records, run_id="health-run")["modules"][0]

    assert module["collapse"] is True
    assert module["severity"] == "one_hot"
    assert module["one_hot"] is True
    assert module["dominant_expert"] == 0
    assert module["top1_share"] == pytest.approx(1.0)
    assert module["gini"] == pytest.approx(15 / 16)
    assert module["gini_matches_one_hot"] is True


def test_diagnose_uses_worst_record_for_collapse() -> None:
    records = [
        RoutingRecord.from_json(_line("moe", "model.8.routing", [1.0, 1.0, 1.0, 1.0], module_type="MoE", paradigm="discrete_selection")),
        RoutingRecord.from_json(_line("moe", "model.8.routing", [9.0, 1.0, 0.0, 0.0], module_type="MoE", paradigm="discrete_selection")),
    ]

    module = diagnose_records(records)["modules"][0]

    assert module["records"] == 2
    assert module["top1_share"] == pytest.approx(0.9)
    assert module["collapse"] is True
    assert module["severity"] == "concentrated"


def test_invalid_inputs_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        diagnose_records([])
    with pytest.raises(FileNotFoundError):
        check_run(tmp_path / "missing-run")


def test_classify_boundary_is_inclusive() -> None:
    at_threshold = classify_module(
        num_experts=4, gini=0.5, entropy_normalized=0.5, top1_share=COLLAPSE_SHARE_THRESHOLD
    )
    below_threshold = classify_module(
        num_experts=4, gini=0.5, entropy_normalized=0.5, top1_share=COLLAPSE_SHARE_THRESHOLD - 1e-9
    )
    assert at_threshold["collapse"] is True
    assert below_threshold["collapse"] is False