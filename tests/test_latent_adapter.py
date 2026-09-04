"""Unit tests for the Latent -> RoutingRecord v1 adapter (E3 P0)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.latent_adapter import LatentAdapter  # noqa: E402
from scripts.routing_record import RoutingRecord, SCHEMA_VERSION  # noqa: E402


def _latent_snapshot() -> dict:
    import torch

    return {
        "family": "latent",
        "num_experts": 3,
        "top_k": 3,
        "training_top_k": 3,
        "inference_top_k": 2,
        "configured_top_k": 3,
        "executed_experts": 3,
        "active_experts_per_sample": torch.tensor([3, 3]),
        "mean_active_experts_per_sample": 3.0,
        "batch_expert_union": 3,
        "kernel_calls": 3,
        "mean_router_probs": torch.tensor([0.5, 0.3, 0.2]),
        "expert_usage": torch.tensor([0.5, 0.3, 0.2]),
        "mean_router_logits": torch.tensor([1.0, 0.0, -1.0]),
        "entropy": torch.tensor(0.9),
        "balance_loss": 0.02,
        "z_loss": 0.01,
        "aux_loss": torch.tensor(0.0123),
        "temperature": 1.0,
        "noise_std": 0.0,
        "router_init_std": 0.02,
        "identity_cold_start": False,
        "residual_gain_magnitude": 0.5,
        "router_output_head_magnitude": 0.7,
        "router_aux_gradient_enabled": True,
        "ddp_balance_synced": False,
        "dispatch_policy": "dense",
        "finite": True,
        "routing_axis": "expert",
        "value_fusion_mode": "router_only",
        "value_fusion_weights": torch.tensor([1.0, 0.0]),
        "inference_calibrated": True,
        "inference_calibration_error": 0.0002,
        "inference_calibration_batches": 32,
        "inference_calibration_tolerance": 0.001,
        "extra_tracking": {"untracked": 1},
    }


def _build(snapshot: dict, **kwargs) -> RoutingRecord:
    adapter = LatentAdapter()
    return adapter.to_record(
        snapshot,
        run_id=kwargs.pop("run_id", "e3-smoke-003"),
        captured_at=kwargs.pop("captured_at", "2026-09-03T12:00:00+08:00"),
        module_name=kwargs.pop("module_name", "model.23"),
        module_type=kwargs.pop("module_type", "LatentMixture"),
        step=kwargs.pop("step", 7),
        training=kwargs.pop("training", True),
    )


def test_latent_mixture_snapshot_maps_all_schema_fields() -> None:
    record = _build(_latent_snapshot())
    assert record.schema_version == SCHEMA_VERSION
    assert record.family == "latent"
    assert record.routing_paradigm == "continuous_fusion"
    assert record.module.name == "model.23"
    assert record.module.type == "LatentMixture"
    assert record.step == 7
    assert record.training is True

    routing = record.routing
    assert routing.num_experts == 3
    assert routing.top_k == 3
    assert list(routing.expert_usage) == pytest.approx([0.5, 0.3, 0.2])
    assert routing.normalized_expert_load == pytest.approx([0.5, 0.3, 0.2])
    assert routing.routing_entropy_nats == pytest.approx(1.0296530140645735, abs=1e-4)
    assert 0.0 < routing.routing_entropy_normalized <= 1.0
    assert routing.load_gini == pytest.approx(0.2, abs=1e-4)
    assert routing.dominant_expert == 0
    assert routing.dominant_expert_share == pytest.approx(0.5)
    assert routing.mean_mixing_weights == pytest.approx([0.5, 0.3, 0.2])

    assert record.aux_loss.value == pytest.approx(0.0123)
    assert record.aux_loss.finite is True
    assert record.usage_scope == "rank_local"
    assert record.global_usage_available is False

    assert set(record.family_data) == {
        "configured_top_k",
        "training_top_k",
        "inference_top_k",
        "value_fusion_mode",
        "value_fusion_weights",
        "inference_calibrated",
        "inference_calibration_error",
        "inference_calibration_batches",
        "inference_calibration_tolerance",
        "routing_axis",
        "dispatch_policy",
        "executed_experts",
        "mean_active_experts_per_sample",
        "batch_expert_union",
        "kernel_calls",
        "balance_loss",
        "z_loss",
    }
    assert record.family_data["configured_top_k"] == 3
    assert record.family_data["training_top_k"] == 3
    assert record.family_data["inference_top_k"] == 2
    assert record.family_data["routing_axis"] == "expert"
    assert record.family_data["value_fusion_mode"] == "router_only"
    assert record.family_data["value_fusion_weights"] == pytest.approx([1.0, 0.0])
    assert record.family_data["inference_calibrated"] is True
    assert record.family_data["inference_calibration_error"] == pytest.approx(0.0002)
    assert record.family_data["inference_calibration_batches"] == 32
    assert record.family_data["inference_calibration_tolerance"] == pytest.approx(0.001)
    assert record.family_data["dispatch_policy"] == "dense"
    assert record.family_data["executed_experts"] == 3
    assert record.family_data["mean_active_experts_per_sample"] == pytest.approx(3.0)
    assert record.family_data["batch_expert_union"] == 3
    assert record.family_data["kernel_calls"] == 3
    assert record.family_data["balance_loss"] == pytest.approx(0.02)
    assert record.family_data["z_loss"] == pytest.approx(0.01)

    excluded = {
        "family",
        "entropy",
        "mean_router_logits",
        "temperature",
        "noise_std",
        "router_init_std",
        "identity_cold_start",
        "residual_gain_magnitude",
        "router_output_head_magnitude",
        "router_aux_gradient_enabled",
        "ddp_balance_synced",
        "active_experts_per_sample",
        "extra_tracking",
    }
    assert not set(record.family_data) & excluded

    assert list(record.source_snapshot_keys) == sorted(_latent_snapshot().keys())


def test_scale_expert_axis_family_data() -> None:
    import torch

    snapshot = _latent_snapshot()
    snapshot.update(
        {
            "routing_axis": "scale_expert",
            "num_scales": 2,
            "scale_mean_probs": torch.tensor([[0.6, 0.3, 0.1], [0.4, 0.3, 0.3]]),
            "mean_router_probs": torch.tensor([0.5, 0.3, 0.2]),
            "expert_usage": torch.tensor([0.5, 0.3, 0.2]),
        }
    )
    record = _build(snapshot, module_type="MultiScaleLatentMixture")
    assert record.family_data["routing_axis"] == "scale_expert"
    assert record.family_data["num_scales"] == 2
    scale_mean_probs = record.family_data["scale_mean_probs"]
    assert scale_mean_probs[0] == pytest.approx([0.6, 0.3, 0.1])
    assert scale_mean_probs[1] == pytest.approx([0.4, 0.3, 0.3])
    assert record.routing.num_experts == 3
    assert record.routing.mean_mixing_weights == pytest.approx([0.5, 0.3, 0.2])


def test_missing_required_field_rejected() -> None:
    snapshot = _latent_snapshot()
    del snapshot["expert_usage"]
    with pytest.raises(ValueError, match="expert_usage"):
        _build(snapshot)

    snapshot = _latent_snapshot()
    snapshot["expert_usage"] = None
    with pytest.raises(ValueError, match="expert_usage"):
        _build(snapshot)

    snapshot = _latent_snapshot()
    del snapshot["num_experts"]
    with pytest.raises(ValueError, match="num_experts"):
        _build(snapshot)

    snapshot = _latent_snapshot()
    del snapshot["top_k"]
    with pytest.raises(ValueError, match="top_k"):
        _build(snapshot)

    with pytest.raises(ValueError, match="must be a mapping"):
        _build([1, 2, 3])


def test_optional_missing_fields_use_v1_defaults() -> None:
    snapshot = {"num_experts": 3, "top_k": 3, "expert_usage": [1.0, 1.0, 1.0]}
    record = _build(snapshot, step=None, training=None)
    assert record.step is None
    assert record.training is None
    assert record.routing.mean_mixing_weights is None
    assert record.aux_loss.value is None
    assert record.aux_loss.finite is None
    assert record.family_data == {}
    assert record.usage_scope == "rank_local"
    assert record.global_usage_available is False
    assert list(record.source_snapshot_keys) == ["expert_usage", "num_experts", "top_k"]
    assert record.routing.routing_entropy_nats == pytest.approx(math.log(3.0), rel=1e-5)
    assert record.routing.routing_entropy_normalized == pytest.approx(1.0, abs=1e-4)


def test_aux_finite_flag_fallback_and_override() -> None:
    snapshot = {"num_experts": 3, "top_k": 3, "expert_usage": [1.0, 1.0, 1.0], "aux_loss": 0.5}
    record = _build(snapshot)
    assert record.aux_loss.value == pytest.approx(0.5)
    assert record.aux_loss.finite is True

    snapshot["finite"] = False
    record = _build(snapshot)
    assert record.aux_loss.value == pytest.approx(0.5)
    assert record.aux_loss.finite is False


def test_boundary_uniform_dense_usage() -> None:
    snapshot = {
        "num_experts": 4,
        "top_k": 4,
        "expert_usage": [1.0, 1.0, 1.0, 1.0],
        "aux_loss": 0.0,
        "finite": True,
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
    snapshot = _latent_snapshot()
    snapshot["top_k"] = 0
    with pytest.raises(ValueError, match="top_k"):
        _build(snapshot)

    snapshot = _latent_snapshot()
    snapshot["top_k"] = 4
    with pytest.raises(ValueError, match="top_k"):
        _build(snapshot)

    snapshot = _latent_snapshot()
    snapshot["num_experts"] = 0
    snapshot["expert_usage"] = []
    with pytest.raises(ValueError, match="num_experts"):
        _build(snapshot)

    snapshot = _latent_snapshot()
    snapshot["expert_usage"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="length"):
        _build(snapshot)

    snapshot = _latent_snapshot()
    snapshot["expert_usage"] = "oops"
    with pytest.raises(ValueError, match="expert_usage"):
        _build(snapshot)


def test_v1_output_structure_stable_round_trip() -> None:
    record = _build(_latent_snapshot())
    payload = record.to_dict()
    assert set(payload) == {
        "schema_version",
        "run_id",
        "captured_at",
        "family",
        "routing_paradigm",
        "module",
        "step",
        "training",
        "routing",
        "aux_loss",
        "usage_scope",
        "global_usage_available",
        "family_data",
        "source_snapshot_keys",
    }
    assert set(payload["routing"]) == {
        "num_experts",
        "top_k",
        "expert_usage",
        "normalized_expert_load",
        "routing_entropy_nats",
        "routing_entropy_normalized",
        "load_gini",
        "dominant_expert",
        "dominant_expert_share",
        "mean_mixing_weights",
    }
    assert set(payload["module"]) == {"name", "type"}
    assert all(isinstance(value, float) for value in payload["routing"]["expert_usage"])
    assert all(
        isinstance(value, float) for value in payload["family_data"]["value_fusion_weights"]
    )
    assert RoutingRecord.from_dict(payload) == record
    assert json.loads(record.to_json()) == payload
    assert RoutingRecord.from_json(record.to_json()) == record