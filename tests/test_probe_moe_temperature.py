"""Tests for the MoE temperature probe summary helpers.

``scripts/probe_moe_temperature.py`` needs a deployed YOLO-Master checkout to
run, so these tests cover its pure comparison surface: a wrong delta, a missing
arm, or an order-dependent digest must fail here rather than inside an evidence
run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_moe_temperature import (
    DEFAULT_FACTOR,
    DEFAULT_MIN_TEMP,
    FIELDS,
    aggregate_deltas,
    arm_digest,
    metric_delta,
    summarize_layers,
)


def _row(share: float, nats: float, normalized: float, gini: float, *, expert: int = 0) -> dict:
    return {
        "module_type": "DetailAwareLowRankHybridAdaptiveGateMoE",
        "num_experts": 4,
        "top_k": 2,
        "dominant_expert": expert,
        "usage": [share, 1.0 - share, 0.0, 0.0],
        "top1_share": share,
        "entropy_nats": nats,
        "entropy_normalized": normalized,
        "gini": gini,
    }


def test_documented_run_parameters_match_script_defaults() -> None:
    # docs/moe-temperature-probe.md records these exact values.
    assert DEFAULT_FACTOR == 2.0
    assert DEFAULT_MIN_TEMP == 0.3


def test_metric_delta_is_absolute_after_minus_before() -> None:
    delta = metric_delta(_row(0.5, 0.6, 0.3, 0.2), _row(0.4, 0.65, 0.35, 0.1))
    assert delta == pytest.approx(
        {"top1_share": -0.1, "entropy_nats": 0.05, "entropy_normalized": 0.05, "gini": -0.1}
    )
    assert set(delta) == set(FIELDS)


def test_metric_delta_fails_loudly_on_missing_field() -> None:
    before = _row(0.5, 0.6, 0.3, 0.2)
    before.pop("gini")
    with pytest.raises(ValueError, match="gini"):
        metric_delta(before, _row(0.4, 0.65, 0.35, 0.1))


def test_summarize_layers_fails_loudly_on_missing_layer() -> None:
    with pytest.raises(ValueError, match="model.8"):
        summarize_layers({"model.8": _row(0.5, 0.6, 0.3, 0.2)}, {})


def test_summarize_layers_records_before_after_delta_and_dominant_expert() -> None:
    baseline = {"model.8": _row(0.536595, 0.690466, 0.332044, 0.759149, expert=1)}
    intervention = {"model.8": _row(0.518322, 0.692476, 0.333010, 0.754580, expert=1)}
    layer = summarize_layers(baseline, intervention)["model.8"]
    assert layer["baseline"]["top1_share"] == pytest.approx(0.536595)
    assert layer["factor_2_0"]["gini"] == pytest.approx(0.754580)
    assert layer["delta"]["top1_share"] == pytest.approx(-0.018273)
    assert layer["dominant_expert"] == {"baseline": 1, "factor_2_0": 1}
    assert layer["dominant_expert_changed"] is False


def test_aggregate_deltas_counts_changed_layers_and_means_over_all_layers() -> None:
    layers = summarize_layers(
        {"model.5": _row(1.0, 0.0, 0.0, 0.75), "model.8": _row(0.5, 0.6, 0.3, 0.2)},
        {"model.5": _row(1.0, 0.0, 0.0, 0.75), "model.8": _row(0.4, 0.7, 0.4, 0.1)},
    )
    aggregate = aggregate_deltas(layers)
    assert aggregate["moe_layers"] == 2
    # the unchanged layer still counts in the denominator
    assert aggregate["changed_layers"]["top1_share"] == 1
    assert aggregate["mean_delta"]["top1_share"] == pytest.approx(-0.05)
    assert aggregate["changed_layers"]["gini"] == 1
    assert aggregate["dominant_expert_changed_layers"] == 0


def test_aggregate_deltas_handles_no_layers() -> None:
    aggregate = aggregate_deltas({})
    assert aggregate["moe_layers"] == 0
    assert aggregate["mean_delta"] == dict.fromkeys(FIELDS, 0.0)


def test_arm_digest_is_order_independent_and_value_sensitive() -> None:
    first = {"model.5": _row(1.0, 0.0, 0.0, 0.75), "model.8": _row(0.5, 0.6, 0.3, 0.2)}
    reordered = {"model.8": _row(0.5, 0.6, 0.3, 0.2), "model.5": _row(1.0, 0.0, 0.0, 0.75)}
    assert arm_digest(first) == arm_digest(reordered)

    changed = {"model.5": _row(1.0, 0.0, 0.0, 0.75), "model.8": _row(0.4, 0.6, 0.3, 0.2)}
    assert arm_digest(first) != arm_digest(changed)