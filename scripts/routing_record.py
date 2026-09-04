"""E3 RoutingRecord v1: data model, validation, and JSON/JSONL round trips.

Schema version: ``e3-routing/v1``.

Scope (P0-B step 1):
- Canonical routing record shared across MoE / MoT / Latent.
- No family adapters yet; later adapters map upstream ``last_routing_snapshot``
  payloads into this record.
- Stdlib only (``dataclasses`` + ``json``); no third-party dependencies.

Design rules enforced here:
- ``family_data`` is always a plain dict and is never auto-populated from a
  whole upstream snapshot. Adapters must place scalar/list family fields
  explicitly and keep ``source_snapshot_keys`` as the drift-detection trail.
- Derived metrics (entropy / gini / dominant share) are validated here but are
  expected to be produced by upstream helpers at the adapter layer; this module
  does not reimplement routing formulas.
- ``schema_version`` is pinned to ``e3-routing/v1`` on both construction and
  deserialization.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "e3-routing/v1"

# Finite enums. Paradigm tokens follow the E3 field-dictionary vocabulary:
# MoE = discrete expert selection, MoT = scene-conditioned routing,
# Latent = continuous fusion over latent experts/scales.
FAMILIES = frozenset({"moe", "mot", "latent"})
ROUTING_PARADIGMS = frozenset({"discrete_selection", "scene_conditioned", "continuous_fusion"})
USAGE_SCOPES = frozenset({"rank_local", "global"})

_REQUIRED_RECORD_KEYS = ("run_id", "captured_at", "family", "routing_paradigm", "module", "routing")
_REQUIRED_ROUTING_KEYS = (
    "num_experts",
    "top_k",
    "expert_usage",
    "normalized_expert_load",
    "routing_entropy_nats",
    "routing_entropy_normalized",
    "load_gini",
    "dominant_expert",
    "dominant_expert_share",
)
_ALL_ROUTING_KEYS = _REQUIRED_ROUTING_KEYS + ("mean_mixing_weights",)
_BOUND_EPS = 1e-9
_LOAD_SUM_TOLERANCE = 1e-6


def _missing(payload: Mapping[str, Any], required: Sequence[str]) -> list[str]:
    return sorted(key for key in required if key not in payload)


def _require_int(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}")
    if maximum is not None and out > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {out}")
    return out


def _require_finite(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}")
    return out


def _coerce_float_sequence(value: Any, name: str) -> tuple[float, ...]:
    if value is None:
        raise ValueError(f"{name} is required")
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise ValueError(f"{name} must be a sequence of numbers, got {type(value).__name__}")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numbers, got {value!r}") from exc


def _coerce_str_sequence(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings, got a single string")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} must contain only strings, got {item!r}")
        out.append(item)
    return tuple(out)


@dataclass(frozen=True)
class ModuleRef:
    """Source module identity: named-modules path and runtime class name."""

    name: str
    type: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("module.name is required and must be a non-empty string")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("module.type is required and must be a non-empty string")


@dataclass(frozen=True)
class AuxLoss:
    """Optional auxiliary-loss scalar and its finiteness flag."""

    value: float | None = None
    finite: bool | None = None

    def __post_init__(self) -> None:
        if self.value is not None:
            object.__setattr__(self, "value", _require_finite(self.value, "aux_loss.value"))
        if self.finite is not None and not isinstance(self.finite, bool):
            raise ValueError("aux_loss.finite must be a bool or null")


@dataclass(frozen=True)
class RoutingStats:
    """Family-agnostic routing statistics (raw usage + derived metrics)."""

    num_experts: int
    top_k: int
    expert_usage: Sequence[float]
    normalized_expert_load: Sequence[float]
    routing_entropy_nats: float
    routing_entropy_normalized: float
    load_gini: float
    dominant_expert: int
    dominant_expert_share: float
    mean_mixing_weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        num_experts = _require_int(self.num_experts, "routing.num_experts", minimum=1)
        top_k = _require_int(self.top_k, "routing.top_k", minimum=1, maximum=num_experts)
        usage = _coerce_float_sequence(self.expert_usage, "routing.expert_usage")
        load = _coerce_float_sequence(self.normalized_expert_load, "routing.normalized_expert_load")
        if len(usage) != num_experts:
            raise ValueError(f"routing.expert_usage length {len(usage)} != num_experts {num_experts}")
        if len(load) != num_experts:
            raise ValueError(f"routing.normalized_expert_load length {len(load)} != num_experts {num_experts}")
        if abs(sum(load) - 1.0) > _LOAD_SUM_TOLERANCE:
            raise ValueError(f"routing.normalized_expert_load must sum to ~1.0, got {sum(load)}")
        entropy = _require_finite(self.routing_entropy_nats, "routing.routing_entropy_nats")
        normalized = _require_finite(self.routing_entropy_normalized, "routing.routing_entropy_normalized")
        gini = _require_finite(self.load_gini, "routing.load_gini")
        if normalized > 1.0 + _BOUND_EPS:
            raise ValueError(f"routing.routing_entropy_normalized must be <= 1, got {normalized}")
        if gini > 1.0 + _BOUND_EPS:
            raise ValueError(f"routing.load_gini must be <= 1, got {gini}")
        dominant = _require_int(self.dominant_expert, "routing.dominant_expert", minimum=-1, maximum=num_experts - 1)
        share = _require_finite(self.dominant_expert_share, "routing.dominant_expert_share")
        if share > 1.0 + _BOUND_EPS:
            raise ValueError(f"routing.dominant_expert_share must be <= 1, got {share}")
        if self.mean_mixing_weights is not None:
            mixing = _coerce_float_sequence(self.mean_mixing_weights, "routing.mean_mixing_weights")
            if len(mixing) != num_experts:
                raise ValueError(f"routing.mean_mixing_weights length {len(mixing)} != num_experts {num_experts}")
            object.__setattr__(self, "mean_mixing_weights", mixing)
        object.__setattr__(self, "num_experts", num_experts)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "expert_usage", usage)
        object.__setattr__(self, "normalized_expert_load", load)
        object.__setattr__(self, "routing_entropy_nats", entropy)
        object.__setattr__(self, "routing_entropy_normalized", normalized)
        object.__setattr__(self, "load_gini", gini)
        object.__setattr__(self, "dominant_expert", dominant)
        object.__setattr__(self, "dominant_expert_share", share)


@dataclass(frozen=True)
class RoutingRecord:
    """One canonical E3 routing observation (schema ``e3-routing/v1``)."""

    run_id: str
    captured_at: str
    family: str
    routing_paradigm: str
    module: ModuleRef
    routing: RoutingStats
    schema_version: str = SCHEMA_VERSION
    step: int | None = None
    training: bool | None = None
    aux_loss: AuxLoss = field(default_factory=AuxLoss)
    usage_scope: str = "rank_local"
    global_usage_available: bool = False
    family_data: dict[str, Any] = field(default_factory=dict)
    source_snapshot_keys: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id is required and must be a non-empty string")
        if not isinstance(self.captured_at, str) or not self.captured_at:
            raise ValueError("captured_at is required and must be a non-empty string")
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {self.family!r}")
        if self.routing_paradigm not in ROUTING_PARADIGMS:
            raise ValueError(
                f"routing_paradigm must be one of {sorted(ROUTING_PARADIGMS)}, got {self.routing_paradigm!r}"
            )
        if not isinstance(self.module, ModuleRef):
            raise ValueError("module must be a ModuleRef")
        if not isinstance(self.routing, RoutingStats):
            raise ValueError("routing must be a RoutingStats")
        if not isinstance(self.aux_loss, AuxLoss):
            raise ValueError("aux_loss must be an AuxLoss")
        if self.step is not None:
            object.__setattr__(self, "step", _require_int(self.step, "step", minimum=0))
        if self.training is not None and not isinstance(self.training, bool):
            raise ValueError("training must be a bool or null")
        if self.usage_scope not in USAGE_SCOPES:
            raise ValueError(f"usage_scope must be one of {sorted(USAGE_SCOPES)}, got {self.usage_scope!r}")
        if not isinstance(self.global_usage_available, bool):
            raise ValueError("global_usage_available must be a bool")
        if not isinstance(self.family_data, dict):
            raise ValueError("family_data must be a dict (never auto-copied from a snapshot)")
        object.__setattr__(self, "source_snapshot_keys", _coerce_str_sequence(self.source_snapshot_keys, "source_snapshot_keys"))

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (nested objects expanded)."""
        routing = self.routing
        mixing = None if routing.mean_mixing_weights is None else list(routing.mean_mixing_weights)
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "captured_at": self.captured_at,
            "family": self.family,
            "routing_paradigm": self.routing_paradigm,
            "module": {"name": self.module.name, "type": self.module.type},
            "step": self.step,
            "training": self.training,
            "routing": {
                "num_experts": routing.num_experts,
                "top_k": routing.top_k,
                "expert_usage": list(routing.expert_usage),
                "normalized_expert_load": list(routing.normalized_expert_load),
                "routing_entropy_nats": routing.routing_entropy_nats,
                "routing_entropy_normalized": routing.routing_entropy_normalized,
                "load_gini": routing.load_gini,
                "dominant_expert": routing.dominant_expert,
                "dominant_expert_share": routing.dominant_expert_share,
                "mean_mixing_weights": mixing,
            },
            "aux_loss": {"value": self.aux_loss.value, "finite": self.aux_loss.finite},
            "usage_scope": self.usage_scope,
            "global_usage_available": self.global_usage_available,
            "family_data": dict(self.family_data),
            "source_snapshot_keys": list(self.source_snapshot_keys),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoutingRecord":
        """Build a record from a JSON-friendly dict with required-field checks."""
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")
        data = dict(payload)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}, got {data.get('schema_version')!r}")

        missing = _missing(data, _REQUIRED_RECORD_KEYS)
        if missing:
            raise ValueError("missing required field(s): " + ", ".join(missing))

        module_payload = data["module"]
        if not isinstance(module_payload, Mapping):
            raise ValueError("module must be an object with 'name' and 'type'")
        missing_module = _missing(module_payload, ("name", "type"))
        if missing_module:
            raise ValueError("missing required module field(s): " + ", ".join(missing_module))

        routing_payload = data["routing"]
        if not isinstance(routing_payload, Mapping):
            raise ValueError("routing must be an object")
        missing_routing = _missing(routing_payload, _REQUIRED_ROUTING_KEYS)
        if missing_routing:
            raise ValueError("missing required routing field(s): " + ", ".join(missing_routing))
        routing_kwargs = {key: routing_payload[key] for key in _ALL_ROUTING_KEYS if key in routing_payload}

        aux_payload = data.get("aux_loss")
        if aux_payload is None:
            aux = AuxLoss()
        elif isinstance(aux_payload, Mapping):
            aux = AuxLoss(value=aux_payload.get("value"), finite=aux_payload.get("finite"))
        else:
            raise ValueError("aux_loss must be an object or null")

        return cls(
            run_id=data["run_id"],
            captured_at=data["captured_at"],
            family=data["family"],
            routing_paradigm=data["routing_paradigm"],
            module=ModuleRef(name=module_payload["name"], type=module_payload["type"]),
            routing=RoutingStats(**routing_kwargs),
            step=data.get("step"),
            training=data.get("training"),
            aux_loss=aux,
            usage_scope=data.get("usage_scope", "rank_local"),
            global_usage_available=data.get("global_usage_available", False),
            family_data=data.get("family_data", {}),
            source_snapshot_keys=data.get("source_snapshot_keys", ()),
        )

    def to_json(self) -> str:
        """Serialize to one compact JSON line (JSONL-ready)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "RoutingRecord":
        """Deserialize one JSONL line produced by ``to_json``."""
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON record line: {line[:120]!r}") from exc
        return cls.from_dict(payload)


__all__ = [
    "AuxLoss",
    "FAMILIES",
    "ModuleRef",
    "ROUTING_PARADIGMS",
    "RoutingRecord",
    "RoutingStats",
    "SCHEMA_VERSION",
    "USAGE_SCOPES",
]
