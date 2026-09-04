"""Contract tests for the E3 RoutingRecord v1 data model (P0-B step 1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.routing_record import SCHEMA_VERSION, RoutingRecord  # noqa: E402


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "e3-smoke-001",
        "captured_at": "2026-09-03T10:00:00+08:00",
        "family": "moe",
        "routing_paradigm": "discrete_selection",
        "module": {"name": "model.5.routing", "type": "TopKMoERouter"},
        "step": 42,
        "training": True,
        "routing": {
            "num_experts": 4,
            "top_k": 2,
            "expert_usage": [3.0, 1.0, 0.0, 0.0],
            "normalized_expert_load": [0.75, 0.25, 0.0, 0.0],
            "routing_entropy_nats": 0.5623351446188083,
            "routing_entropy_normalized": 0.4056390622295663,
            "load_gini": 0.625,
            "dominant_expert": 0,
            "dominant_expert_share": 0.75,
            "mean_mixing_weights": [0.8, 0.2, 0.0, 0.0],
        },
        "aux_loss": {"value": 0.0123, "finite": True},
        "usage_scope": "rank_local",
        "global_usage_available": False,
        "family_data": {"scene_aware": False, "dispatch_policy": "dense"},
        "source_snapshot_keys": ["num_experts", "top_k", "expert_usage"],
    }


def test_dict_round_trip() -> None:
    payload = _valid_payload()
    record = RoutingRecord.from_dict(payload)
    assert record.schema_version == "e3-routing/v1"
    assert record.to_dict() == payload
    assert isinstance(record.family_data, dict)
    assert record.family_data == {"scene_aware": False, "dispatch_policy": "dense"}


def test_json_round_trip() -> None:
    payload = _valid_payload()
    record = RoutingRecord.from_dict(payload)
    line = record.to_json()
    assert json.loads(line) == payload
    assert RoutingRecord.from_json(line) == record


def test_missing_required_fields_rejected() -> None:
    payload = _valid_payload()
    del payload["run_id"]
    with pytest.raises(ValueError, match="run_id"):
        RoutingRecord.from_dict(payload)

    payload = _valid_payload()
    del payload["routing"]["load_gini"]
    with pytest.raises(ValueError, match="load_gini"):
        RoutingRecord.from_dict(payload)


def test_invalid_family_rejected() -> None:
    payload = _valid_payload()
    payload["family"] = "gpt_moe"
    with pytest.raises(ValueError, match="family"):
        RoutingRecord.from_dict(payload)

    payload = _valid_payload()
    payload["routing_paradigm"] = "topk"
    with pytest.raises(ValueError, match="routing_paradigm"):
        RoutingRecord.from_dict(payload)


def test_unknown_schema_version_rejected() -> None:
    payload = _valid_payload()
    payload["schema_version"] = "e3-routing/v2"
    with pytest.raises(ValueError, match="schema_version"):
        RoutingRecord.from_dict(payload)


def test_family_data_must_be_dict() -> None:
    payload = _valid_payload()
    payload["family_data"] = "not-a-dict"
    with pytest.raises(ValueError, match="family_data"):
        RoutingRecord.from_dict(payload)
