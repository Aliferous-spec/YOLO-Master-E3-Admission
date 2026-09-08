"""MoT routing snapshot -> E3 RoutingRecord v1 adapter (E3 P0 continuation).

Scope:
- Converts an upstream MoT ``last_routing_snapshot`` dict (MoTBlock or
  MoT wrapper aggregation) into a validated ``e3-routing/v1``
  ``RoutingRecord`` with family="mot" and
  routing_paradigm="scene_conditioned".
- Uses only routing diagnostics that the upstream MoT snapshot actually
  exposes. Fields MoT does not provide are left to their v1 default/None.
- The v1 schema is reused as-is; this module does not change it.

Metric policy (mirrors ``moe_adapter``):
- Derived metrics (entropy, gini, usage scope) are reused from the upstream
  helper ``ultralytics.nn.modules.routing_protocol.global_routing_metrics``;
  no routing formula is reimplemented here.
- Simple arithmetic upstream does not expose (sum-1 shares, entropy / ln(E) +
  clamp, argmax) lives once in ``RoutingAdapterBase``.
- ``family_data`` is whitelisted to MoT scene/dispatch diagnostics that the
  upstream snapshot carries; the whole snapshot is never copied.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.routing_adapter_base import RoutingAdapterBase


class MoTAdapter(RoutingAdapterBase):
    """Adapt one MoT routing snapshot into a v1 RoutingRecord."""

    FAMILY = "mot"
    ROUTING_PARADIGM = "scene_conditioned"
    FAMILY_DATA_WHITELIST = (
        "scene_aware",
        "scene_stats",
        "scene_bias",
        "scene_inference_mode",
        "scene_aware_applied",
        "scene_bypass_reason",
        "scene_consistency_loss",
        "dispatch",
    )

    @staticmethod
    def _aux_finite(finite_diagnostics: Any) -> bool | None:
        """Read ``aux_loss_finite`` from a MoT diagnostics payload.

        MoTBlock stores a mapping; the wrapper aggregation stores a list of
        per-child mappings. Returns None when no usable flag exists.
        """
        if isinstance(finite_diagnostics, Mapping):
            flag = finite_diagnostics.get("aux_loss_finite")
            return flag if isinstance(flag, bool) else None
        if isinstance(finite_diagnostics, list):
            flags = [
                item.get("aux_loss_finite")
                for item in finite_diagnostics
                if isinstance(item, Mapping) and isinstance(item.get("aux_loss_finite"), bool)
            ]
            if flags:
                return bool(all(flags))
        return None

    def _snapshot_finite(self, plain: Mapping[str, Any]) -> bool | None:
        """Wire the MoT finite-diagnostics payload into the shared hook."""
        return self._aux_finite(plain.get("finite_diagnostics"))


__all__ = ["MoTAdapter"]