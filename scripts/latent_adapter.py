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
- Simple arithmetic upstream does not expose (sum-1 shares, entropy / ln(E),
  argmax) stays local.
- ``family_data`` is whitelisted to Latent diagnostics carried by the upstream
  snapshot: the train/inference top-k split, value-fusion and inference
  calibration state, scale-expert routing fields, dispatch/execution counters,
  and the aux-loss decomposition (balance / z loss). The whole snapshot is
  never copied into it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ultralytics.nn.modules.routing_protocol import global_routing_metrics

from scripts.routing_record import AuxLoss, ModuleRef, RoutingRecord, RoutingStats


class LatentAdapter:
    """Adapt one Latent routing snapshot into a v1 RoutingRecord."""

    FAMILY = "latent"
    ROUTING_PARADIGM = "continuous_fusion"
    REQUIRED_SNAPSHOT_KEYS = ("num_experts", "top_k", "expert_usage")
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

    @staticmethod
    def _to_plain(value: Any) -> Any:
        """Convert tensors / numpy scalars into JSON-safe plain values."""
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu()
            if getattr(value, "ndim", 0) == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): LatentAdapter._to_plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [LatentAdapter._to_plain(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        return value

    @staticmethod
    def _float_list(value: Any, name: str) -> list[float]:
        if value is None:
            raise ValueError(f"{name} is required")
        if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
            raise ValueError(f"{name} must be a sequence of numbers, got {type(value).__name__}")
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only numbers, got {value!r}") from exc

    @staticmethod
    def _snapshot_finite(plain: Mapping[str, Any]) -> bool | None:
        """Read the flat ``finite`` flag published by the Latent snapshot.

        Upstream Latent snapshots summarize finiteness as a single boolean
        (``routing_finite_diagnostics(...).get("all_finite")``). Returns None
        when the flag is absent or not an explicit bool.
        """
        flag = plain.get("finite")
        return flag if isinstance(flag, bool) else None

    def to_record(
        self,
        snapshot: Mapping[str, Any],
        *,
        run_id: str,
        captured_at: str,
        module_name: str,
        module_type: str,
        step: int | None = None,
        training: bool | None = None,
    ) -> RoutingRecord:
        """Build one validated RoutingRecord from an upstream Latent snapshot."""
        if not isinstance(snapshot, Mapping):
            raise ValueError(f"snapshot must be a mapping, got {type(snapshot).__name__}")
        missing = sorted(key for key in self.REQUIRED_SNAPSHOT_KEYS if key not in snapshot)
        if missing:
            raise ValueError("missing required Latent snapshot field(s): " + ", ".join(missing))

        plain = self._to_plain(dict(snapshot))
        num_experts = int(plain["num_experts"])
        top_k = int(plain["top_k"])

        usage_raw = plain.get("expert_usage")
        if usage_raw is None:
            raise ValueError("Latent snapshot expert_usage is None; no usage vector to record")
        usage = self._float_list(usage_raw, "expert_usage")
        if len(usage) != num_experts:
            raise ValueError(f"expert_usage length {len(usage)} != num_experts {num_experts}")

        # Derived metrics come from the upstream helper; do not duplicate it.
        upstream = global_routing_metrics({"expert_usage": usage})
        entropy_nats = float(upstream["global_entropy"])
        gini = float(upstream["global_gini"])
        global_available = bool(upstream["global_usage_available"])
        usage_scope = (
            str(plain["usage_scope"])
            if isinstance(plain.get("usage_scope"), str)
            else "global" if global_available else "rank_local"
        )

        total = sum(usage)
        shares = [item / total for item in usage] if total > 0.0 else [0.0] * len(usage)
        denom = math.log(num_experts) if num_experts > 1 else 0.0
        normalized_entropy = entropy_nats / denom if denom > 0.0 else 0.0
        normalized_entropy = min(max(normalized_entropy, 0.0), 1.0)
        dominant = max(range(len(shares)), key=shares.__getitem__) if shares else -1
        dominant_share = shares[dominant] if dominant >= 0 else 0.0

        mixing_raw = plain.get("mean_router_probs")
        mixing = None if mixing_raw is None else self._float_list(mixing_raw, "mean_router_probs")

        family_data: dict[str, Any] = {}
        for key in self.FAMILY_DATA_WHITELIST:
            value = plain.get(key)
            if value is not None:
                family_data[key] = value

        aux_value = plain.get("aux_loss")
        finite = self._snapshot_finite(plain)
        if aux_value is not None and finite is None:
            finite = math.isfinite(float(aux_value))
        aux = AuxLoss(value=float(aux_value) if aux_value is not None else None, finite=finite)

        return RoutingRecord(
            run_id=run_id,
            captured_at=captured_at,
            family=self.FAMILY,
            routing_paradigm=self.ROUTING_PARADIGM,
            module=ModuleRef(name=module_name, type=module_type),
            step=step,
            training=training,
            routing=RoutingStats(
                num_experts=num_experts,
                top_k=top_k,
                expert_usage=usage,
                normalized_expert_load=shares,
                routing_entropy_nats=entropy_nats,
                routing_entropy_normalized=normalized_entropy,
                load_gini=gini,
                dominant_expert=dominant,
                dominant_expert_share=dominant_share,
                mean_mixing_weights=mixing,
            ),
            aux_loss=aux,
            usage_scope=usage_scope,
            global_usage_available=global_available,
            family_data=family_data,
            source_snapshot_keys=sorted(snapshot.keys()),
        )


__all__ = ["LatentAdapter"]