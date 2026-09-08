"""Latent routing snapshot -> E3 RoutingRecord v1 adapter (E3 P0 continuation).

Scope:
- Converts an upstream Latent ``last_routing_snapshot`` dict (LatentMixture or
  MultiScaleLatentMixture in ``ultralytics.nn.modules.latent_mixture``) into a
  validated ``e3-routing/v1`` ``RoutingRecord`` with family="latent" and
  routing_paradigm="continuous_fusion".
- Uses only routing diagnostics the upstream Latent snapshot actually exposes.
  Fields Latent does not provide are left to their v1 default/None.
- The v1 schema is reused as-is; this module does not change it. MoE / MoT
  adapters are out of scope and are left untouched.

Metric policy (mirrors ``moe_adapter`` / ``mot_adapter``):
- Derived metrics (entropy, gini, usage scope) are reused from the upstream
  helper ``ultralytics.nn.modules.routing_protocol.global_routing_metrics``;
  no routing formula is reimplemented here.
- Simple arithmetic upstream does not expose (sum-1 shares, entropy / ln(E) +
  clamp, argmax) lives once in ``RoutingAdapterBase``.
- ``family_data`` is whitelisted to Latent diagnostics carried by the upstream
  snapshot: the train/inference top-k split, value-fusion and inference
  calibration state, scale-expert routing fields, dispatch/execution counters,
  and the aux-loss decomposition (balance / z loss). The whole snapshot is
  never copied into it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.routing_adapter_base import RoutingAdapterBase


class LatentAdapter(RoutingAdapterBase):
    """Adapt one Latent routing snapshot into a v1 RoutingRecord."""

    FAMILY = "latent"
    ROUTING_PARADIGM = "continuous_fusion"
    FAMILY_DATA_WHITELIST = (
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
        "num_scales",
        "scale_mean_probs",
        "dispatch_policy",
        "executed_experts",
        "mean_active_experts_per_sample",
        "batch_expert_union",
        "kernel_calls",
        "balance_loss",
        "z_loss",
    )

    def _snapshot_finite(self, plain: Mapping[str, Any]) -> bool | None:
        """Read the flat ``finite`` flag published by the Latent snapshot.

        Upstream Latent snapshots summarize finiteness as a single boolean
        (``routing_finite_diagnostics(...).get("all_finite")``). Returns None
        when the flag is absent or not an explicit bool.
        """
        flag = plain.get("finite")
        return flag if isinstance(flag, bool) else None


__all__ = ["LatentAdapter"]