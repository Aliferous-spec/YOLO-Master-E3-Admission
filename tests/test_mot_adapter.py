"""Unit tests for the MoT -> RoutingRecord v1 adapter (E3 P0)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mot_adapter import MoTAdapter  # noqa: E402
from scripts.routing_record import RoutingRecord, SCHEMA_VERSION  # noqa: E402


def _block_snapshot() -> dict:
    import torch

    return {
        "num_experts": 3,
        "top_k": 2,
        "expert_usage": torch.tensor([2.0, 1.0, 1.0]),
        "mean_router_probs": torch.tensor([0.5, 0.3, 0.2]),
        "aux_loss": torch.tensor(0.03, requires_grad=True),
        "scene_aware": True,
        "scene_stats": torch.tensor([1.0, 2.0]),
        "scene_bias": torch.tensor([0.1, -0.2]),
        "scene_inference_mode": "dynamic",
        "scene_aware_applied": True,
        "scene_bypass_reason": None,
        "scene_consistency_loss": 0.0,
        "finite_diagnostics": {"aux_loss_finite": True},
        "dispatch": {
            "mode": "dense",
            "policy": "scene",
            "mean_experts_per_sample": 2.0,
            "sparsity_ratio": 0.25,
            "experts_per_sample": torch.tensor([2.0, 2.0]),
        },
        "extra_tracking": {"untracked": 1},
    }


def _build(snapshot: dict, **kwargs) -> RoutingRecord:
    adapter = MoTAdapter()
    return adapter.to_record(
        snapshot,
        run_id=kwargs.pop("run_id", "e3-smoke-002"),
        captured_at=kwargs.pop("captured_at", "2026-09-03T11:00:00+08:00"),
        module_name=kwargs.pop("module_name", "model.6.c2fmot"),
        module_type=kwargs.pop("module_type", "MoTBlock"),
        step=kwargs.pop("step", 7),
        training=kwargs.pop("training", True),
    )


def test_block_snapshot_maps_all_schema_fields() -> None:
    record = _build(_block_snapshot())
    assert record.schema_version == SCHEMA_VERSION
    assert record.family == "mot"
    assert record.routing_paradigm == "scene_conditioned"
    assert record.module.name == "model.6.c2fmot"
    assert record.module.type == "MoTBlock"
    assert record.step == 7
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

    assert record.aux_loss.value == pytest.approx(0.03)
    assert record.aux_loss.finite is True
    assert record.usage_scope == "rank_local"
    assert record.global_usage_available is False

    assert "extra_tracking" not in record.family_data
    assert set(record.family_data) == {
        "scene_aware",
        "scene_stats",
        "scene_bias",
        "scene_inference_mode",
        "scene_aware_applied",
        "scene_consistency_loss",
        "dispatch",
    }
    assert record.family_data["scene_aware"] is True
    assert record.family_data["scene_stats"] == pytest.approx([1.0, 2.0])
    assert record.family_data["scene_bias"] == pytest.approx([0.1, -0.2])
    assert record.family_data["scene_inference_mode"] == "dynamic"
    assert record.family_data["scene_aware_applied"] is True
    assert record.family_data["scene_consistency_loss"] == pytest.approx(0.0)
    dispatch = record.family_data["dispatch"]
    assert dispatch["mode"] == "dense"
    assert dispatch["policy"] == "scene"
    assert dispatch["mean_experts_per_sample"] == pytest.approx(2.0)
    assert dispatch["experts_per_sample"] == pytest.approx([2.0, 2.0])

    assert list(record.source_snapshot_keys) == sorted(_block_snapshot().keys())

    payload = record.to_dict()
    assert json.loads(record.to_json()) == payload
    assert RoutingRecord.from_json(record.to_json()) == record


def test_wrapper_style_aggregated_snapshot() -> None:
    import torch

    snapshot = {
        "num_experts": 3,
        "top_k": 2,
        "expert_usage": torch.tensor([2.0, 1.0, 1.0]),
        "mean_router_probs": torch.tensor([2.0, 1.0, 1.0]),
        "aux_loss": 0.06,
        "finite_diagnostics": [{"aux_loss_finite": True}, {"aux_loss_finite": True}],
        "dispatch": [{"mode": "dense"}, {"mode": "dense"}],
        "scene_inference_mode": "dynamic",
        "scene_aware_applied": [True, False],
        "scene_bypass_reason": [None, "inference_policy_bypass"],
    }
    record = _build(snapshot, module_type="C2fMoT")
    assert record.family_data["scene_inference_mode"] == "dynamic"
    assert record.family_data["scene_aware_applied"] == [True, False]
    assert record.family_data["scene_bypass_reason"] == [None, "inference_policy_bypass"]
    assert record.aux_loss.finite is True
    assert "scene_aware" not in record.family_data
    assert record.aux_loss.value == pytest.approx(0.06)


def test_missing_required_field_rejected() -> None:
    snapshot = _block_snapshot()
    del snapshot["expert_usage"]
    with pytest.raises(ValueError, match="expert_usage"):
        _build(snapshot)

    snapshot = _block_snapshot()
    del snapshot["num_experts"]
    with pytest.raises(ValueError, match="num_experts"):
        _build(snapshot)


def test_boundary_uniform_dense_usage() -> None:
    snapshot = {
        "num_experts": 4,
        "top_k": 4,
        "expert_usage": [1.0, 1.0, 1.0, 1.0],
        "aux_loss": 0.0,
        "finite_diagnostics": {"aux_loss_finite": True},
    }
    record = _build(snapshot, step=None, training=None)
    assert record.step is None
    assert record.training is None
    routing = record.routing
    assert routing.normalized_expert_load == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert routing.routing_entropy_nats == pytest.approx(math.log(4.0), rel=1e-5)
    assert routing.routing_entropy_normalized == pytest.approx(1.0, abs=1e-4)
    assert routing.load_gini == pytest.approx(0.0, abs=1e-6)
    assert routing.dominant_expert_share == pytest.approx(0.25)


def test_boundary_invalid_values_rejected() -> None:
    snapshot = _block_snapshot()
    snapshot["top_k"] = 0
    with pytest.raises(ValueError, match="top_k"):
        _build(snapshot)

    snapshot = _block_snapshot()
    snapshot["expert_usage"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="length"):
        _build(snapshot)
