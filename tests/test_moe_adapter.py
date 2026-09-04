"""Unit tests for the MoE -> RoutingRecord v1 adapter (P0-C step 1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.moe_adapter import MoEAdapter  # noqa: E402
from scripts.routing_record import RoutingRecord, SCHEMA_VERSION  # noqa: E402


def _snapshot() -> dict:
    import torch

    return {
        "num_experts": 3,
        "top_k": 2,
        "expert_usage": torch.tensor([2.0, 1.0, 1.0]),
        "topk_counts": torch.tensor([3.0, 2.0, 1.0]),
        "mean_router_probs": torch.tensor([0.5, 0.3, 0.2]),
        "mean_topk_weight": torch.tensor([0.6, 0.3, 0.1]),
        "aux_loss": torch.tensor(0.0123, requires_grad=True),
        "finite_diagnostics": {"aux_loss_finite": True},
        "extra_tracking": {"x": 1},
    }


def _build(snapshot: dict, **kwargs) -> RoutingRecord:
    adapter = MoEAdapter()
    return adapter.to_record(
        snapshot,
        run_id=kwargs.pop("run_id", "e3-smoke-001"),
        captured_at=kwargs.pop("captured_at", "2026-09-03T10:00:00+08:00"),
        module_name=kwargs.pop("module_name", "model.5.routing"),
        module_type=kwargs.pop("module_type", "TopKMoERouter"),
        step=kwargs.pop("step", 42),
        training=kwargs.pop("training", True),
    )


def test_normal_snapshot_maps_all_schema_fields() -> None:
    record = _build(_snapshot())
    assert record.schema_version == SCHEMA_VERSION
    assert record.family == "moe"
    assert record.routing_paradigm == "discrete_selection"
    assert record.module.name == "model.5.routing"
    assert record.module.type == "TopKMoERouter"
    assert record.step == 42
    assert record.training is True

    routing = record.routing
    assert routing.num_experts == 3
    assert routing.top_k == 2
    assert list(routing.expert_usage) == [2.0, 1.0, 1.0]
    assert routing.normalized_expert_load == pytest.approx([0.5, 0.25, 0.25])
    assert routing.routing_entropy_nats == pytest.approx(1.0397207736968994, abs=1e-4)
    assert 0.0 < routing.routing_entropy_normalized <= 1.0
    assert routing.load_gini == pytest.approx(1.0 / 6.0, abs=1e-4)
    assert routing.dominant_expert == 0
    assert routing.dominant_expert_share == pytest.approx(0.5)
    assert routing.mean_mixing_weights == pytest.approx([0.5, 0.3, 0.2])

    assert record.aux_loss.value == pytest.approx(0.0123)
    assert record.aux_loss.finite is True
    assert record.usage_scope == "rank_local"
    assert record.global_usage_available is False

    # family_data is a whitelist: only MoE-specific keys, never the snapshot.
    assert set(record.family_data) == {"topk_counts", "mean_topk_weight"}
    assert record.family_data["topk_counts"] == [3.0, 2.0, 1.0]
    assert record.family_data["mean_topk_weight"] == pytest.approx([0.6, 0.3, 0.1])
    assert list(record.source_snapshot_keys) == sorted(_snapshot().keys())


def test_tensors_are_converted_to_json_safe_values() -> None:
    record = _build(_snapshot())
    payload = record.to_dict()
    assert all(isinstance(item, float) for item in payload["routing"]["expert_usage"])
    assert all(isinstance(item, float) for item in payload["family_data"]["topk_counts"])
    assert isinstance(payload["aux_loss"]["value"], float)
    json_line = record.to_json()
    assert json.loads(json_line) == payload
    assert RoutingRecord.from_json(json_line) == record


def test_missing_required_snapshot_field_rejected() -> None:
    snapshot = _snapshot()
    del snapshot["expert_usage"]
    with pytest.raises(ValueError, match="expert_usage"):
        _build(snapshot)


def test_none_usage_rejected() -> None:
    snapshot = _snapshot()
    snapshot["expert_usage"] = None
    with pytest.raises(ValueError, match="expert_usage"):
        _build(snapshot)
