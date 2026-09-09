"""Analytic ground-truth anchors for the routing metric chain (P1 rigor).

These tests pin the metric chain to closed-form mathematical values, not to a
reference implementation.  The chain is: upstream ``global_routing_metrics``
(entropy in nats + bounded Gini) and the adapter-derived fields built on top
of it (normalized entropy, dominant expert, sum-to-one shares).  If any layer
drifts, one of the closed forms below fails.

Closed forms used (E experts, shares p_i = usage_i / sum(usage)):

- Shannon entropy in nats: H = -sum(p_i * ln(p_i)), with 0*ln(0) = 0.
- Gini (scale-invariant, sorted ascending x_i, n values):
  G = 2*sum(i*x_i) / (n*sum(x_i)) - (n+1)/n, clamped to [0, 1].
- [1,0,0]        -> H = 0,          G = 2/3      (max skew, 3 experts)
- [1,1,1]        -> H = ln(3),      G = 0        (perfect balance)
- [1,1]          -> H = ln(2),      G = 0
- [2,1,1]        -> H = 1.5*ln(2),  G = 1/6      (shares 0.5/0.25/0.25)
- [3,1]          -> H = .75*ln(4/3)+.25*ln(4), G = 0.25

Anyone can verify these by hand or with a pocket calculator: that is the
point.  No reference implementation is trusted here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.moe_adapter import MoEAdapter  # noqa: E402
from scripts.routing_record import RoutingRecord  # noqa: E402

LN3 = math.log(3.0)
LN2 = math.log(2.0)


def _metrics(usage: list[float]) -> dict:
    from ultralytics.nn.modules.routing_protocol import global_routing_metrics

    return global_routing_metrics({"expert_usage": usage})


# ---------------------------------------------------------------------------
# Layer 1: upstream global_routing_metrics vs closed-form values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("usage", "expected_entropy", "expected_gini"),
    [
        ([1.0, 0.0, 0.0], 0.0, 2.0 / 3.0),
        ([1.0, 1.0, 1.0], LN3, 0.0),
        ([1.0, 1.0], LN2, 0.0),
        ([2.0, 1.0, 1.0], 1.5 * LN2, 1.0 / 6.0),
        ([3.0, 1.0], 0.75 * math.log(4.0 / 3.0) + 0.25 * math.log(4.0), 0.25),
    ],
    ids=["one-hot-3", "uniform-3", "uniform-2", "skewed-0.5-0.25-0.25", "skewed-0.75-0.25"],
)
def test_upstream_metrics_match_closed_form(usage, expected_entropy, expected_gini) -> None:
    result = _metrics(usage)
    assert result["global_entropy"] == pytest.approx(expected_entropy, rel=1e-5)
    assert result["global_gini"] == pytest.approx(expected_gini, rel=1e-5)


def test_gini_and_entropy_are_scale_invariant() -> None:
    """Usage [2,1,1] and its scaled copy must give identical metrics."""
    a = _metrics([2.0, 1.0, 1.0])
    b = _metrics([0.5, 0.25, 0.25])
    assert a["global_entropy"] == pytest.approx(b["global_entropy"], rel=1e-5)
    assert a["global_gini"] == pytest.approx(b["global_gini"], rel=1e-5)


def test_metrics_are_permutation_invariant() -> None:
    """Expert order must not change global metrics."""
    a = _metrics([2.0, 1.0, 1.0])
    b = _metrics([1.0, 2.0, 1.0])
    c = _metrics([1.0, 1.0, 2.0])
    assert a["global_entropy"] == pytest.approx(b["global_entropy"], rel=1e-5)
    assert a["global_entropy"] == pytest.approx(c["global_entropy"], rel=1e-5)
    assert a["global_gini"] == pytest.approx(b["global_gini"], rel=1e-5)
    assert a["global_gini"] == pytest.approx(c["global_gini"], rel=1e-5)


def test_all_zero_usage_degrades_to_zero_metrics() -> None:
    """Degenerate all-zero usage must not NaN/inf; documented as 0/0."""
    result = _metrics([0.0, 0.0, 0.0])
    assert result["global_entropy"] == 0.0
    assert result["global_gini"] == 0.0


def test_negative_usage_is_sanitized_to_zero() -> None:
    """Upstream clamps negatives to 0: [-1, 1] must equal [0, 1]."""
    sanitized = _metrics([-1.0, 1.0])
    reference = _metrics([0.0, 1.0])
    assert sanitized["global_entropy"] == pytest.approx(reference["global_entropy"], abs=1e-9)
    assert sanitized["global_gini"] == pytest.approx(reference["global_gini"], abs=1e-9)


def test_entropy_bounds() -> None:
    """0 <= H <= ln(E) for every closed-form case above.

    float32 accumulation can round ln(3) up by ~2e-8, hence the tolerance.
    """
    for usage in ([1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [3.0, 1.0]):
        entropy = _metrics(usage)["global_entropy"]
        assert 0.0 <= entropy <= math.log(len(usage)) + 1e-6


# ---------------------------------------------------------------------------
# Layer 2: adapter-derived fields vs closed-form values
# ---------------------------------------------------------------------------


def _record(usage: list[float]) -> RoutingRecord:
    import torch

    snapshot = {
        "num_experts": len(usage),
        "top_k": 1,
        "expert_usage": torch.tensor(usage),
    }
    return MoEAdapter().to_record(
        snapshot,
        run_id="analytic-001",
        captured_at="2026-09-09T23:30:00+08:00",
        module_name="model.0.moe",
        module_type="TopKMoERouter",
    )


def test_adapter_normalized_entropy_is_entropy_over_ln_e() -> None:
    record = _record([2.0, 1.0, 1.0])
    # H = 1.5*ln2, E = 3 -> normalized = 1.5*ln2/ln3.
    assert record.routing.routing_entropy_nats == pytest.approx(1.5 * LN2, rel=1e-5)
    assert record.routing.routing_entropy_normalized == pytest.approx(1.5 * LN2 / LN3, rel=1e-5)


def test_adapter_normalized_entropy_extremes() -> None:
    one_hot = _record([1.0, 0.0, 0.0])
    uniform = _record([1.0, 1.0, 1.0])
    assert one_hot.routing.routing_entropy_normalized == pytest.approx(0.0, abs=1e-9)
    assert uniform.routing.routing_entropy_normalized == pytest.approx(1.0, abs=1e-5)


def test_adapter_shares_sum_to_one_and_dominate_correctly() -> None:
    record = _record([2.0, 1.0, 1.0])
    shares = record.routing.normalized_expert_load
    assert shares == pytest.approx([0.5, 0.25, 0.25], abs=1e-6)
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)
    assert record.routing.dominant_expert == 0
    assert record.routing.dominant_expert_share == pytest.approx(0.5, abs=1e-6)


def test_adapter_single_expert_entropy_is_zero_by_definition() -> None:
    """E = 1 -> ln(E) = 0 -> normalized entropy defined as 0, not NaN."""
    record = _record([7.0])
    assert record.routing.routing_entropy_nats == pytest.approx(0.0, abs=1e-9)
    assert record.routing.routing_entropy_normalized == 0.0
