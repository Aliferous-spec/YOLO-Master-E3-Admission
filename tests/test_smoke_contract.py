"""Contract tests for the E3 admission Smoke package."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_e3_smoke import routing_metrics, validate_records  # noqa: E402


def test_uniform_metrics_are_balanced() -> None:
    metrics = routing_metrics([1.0, 1.0, 1.0])
    assert metrics["expert_load_sum"] == pytest.approx(1.0)
    assert metrics["routing_entropy_normalized"] == pytest.approx(1.0)
    assert metrics["load_gini"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["dominant_expert_share"] == pytest.approx(1.0 / 3.0)


def test_one_hot_metrics_are_concentrated() -> None:
    metrics = routing_metrics([1.0, 0.0, 0.0])
    assert metrics["routing_entropy_normalized"] == pytest.approx(0.0)
    assert metrics["load_gini"] == pytest.approx(2.0 / 3.0)
    assert metrics["dominant_expert_share"] == pytest.approx(1.0)


def test_empty_usage_is_safe() -> None:
    metrics = routing_metrics([])
    assert metrics["expert_load_sum"] == pytest.approx(0.0)
    assert metrics["routing_entropy_normalized"] == pytest.approx(0.0)
    assert metrics["load_gini"] == pytest.approx(0.0)


def test_tensor_input_is_accepted() -> None:
    import torch

    metrics = routing_metrics(torch.tensor([0.2, 0.3, 0.5]))
    assert metrics["expert_load_sum"] == pytest.approx(1.0, abs=1e-6)


def test_validate_records_accepts_clean_rows() -> None:
    clean = {
        "expert_load": [1.0],
        "routing_entropy_normalized": 0.0,
        "load_gini": 0.0,
        "dominant_expert_share": 1.0,
    }
    assert validate_records([clean]) == []


def test_validate_records_flags_bad_rows() -> None:
    bad = {
        "expert_load": [1.0, 0.5],
        "routing_entropy_normalized": float("nan"),
        "load_gini": 0.0,
        "dominant_expert_share": 1.0,
    }
    errors = validate_records([bad])
    assert any("load sum" in error for error in errors)
    assert any("non-finite" in error for error in errors)
