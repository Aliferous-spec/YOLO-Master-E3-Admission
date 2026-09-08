"""MoE routing snapshot -> E3 RoutingRecord v1 adapter (P0-C step 1).

Scope:
- Converts an upstream MoE ``last_routing_snapshot`` dict into a validated
  ``e3-routing/v1`` ``RoutingRecord`` with family="moe" and
  routing_paradigm="discrete_selection".
- Only MoE is supported; MoT / Latent adapters are out of scope.

Metric policy:
- Derived routing metrics (entropy, gini, usage scope) are reused from the
  upstream helper ``ultralytics.nn.modules.routing_protocol.global_routing_metrics``.
  This module does NOT reimplement routing formulas.
- Simple arithmetic upstream does not expose (sum-1 share normalization,
  entropy / ln(E) + clamp, argmax) lives once in ``RoutingAdapterBase`` and is
  shared by all family adapters.
- ``family_data`` is whitelisted to MoE-specific keys only; the whole snapshot
  is never copied into it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.routing_adapter_base import RoutingAdapterBase


class MoEAdapter(RoutingAdapterBase):
    """Adapt one MoE routing snapshot into a v1 RoutingRecord."""

    FAMILY = "moe"
    ROUTING_PARADIGM = "discrete_selection"
    FAMILY_DATA_WHITELIST = ("topk_counts", "mean_topk_weight")

    def _snapshot_finite(self, plain: Mapping[str, Any]) -> bool | None:
        """Read ``aux_loss_finite`` from the MoE diagnostics mapping."""
        diagnostics = plain.get("finite_diagnostics")
        if isinstance(diagnostics, Mapping):
            return diagnostics.get("aux_loss_finite")
        return None

    def _family_data_value(self, key: str, value: Any) -> Any:
        """MoE whitelisted fields are numeric sequences -> float lists."""
        return self._float_list(value, key)


__all__ = ["MoEAdapter"]