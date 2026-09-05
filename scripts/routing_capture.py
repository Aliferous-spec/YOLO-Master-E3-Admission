"""E3 P0 wiring: capture real routing snapshots after a forward and persist them.

Pipeline
    run one real forward (caller-supplied, e.g. ``model(tensor)``)
        -> discover routed modules (routed protocol + non-empty snapshot + family)
        -> adapt each snapshot with the existing per-family adapter
        -> persist one validated e3-routing/v1 ``RoutingRecord`` per line
          through the existing ``RoutingRecordWriter``.

Rules
- Schema / adapters / writer are reused unchanged.
- YOLO-Master core forward is never modified: the caller supplies the forward
  callable and snapshots are only read afterwards.
- Only families with an adapter (moe / mot / latent) are recorded; other
  routed families (e.g. moa) are ignored here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from torch import nn
from ultralytics.nn.modules.routing_protocol import is_routed_module

from scripts.latent_adapter import LatentAdapter
from scripts.moe_adapter import MoEAdapter
from scripts.mot_adapter import MoTAdapter
from scripts.routing_record import RoutingRecord
from scripts.routing_record_writer import RoutingRecordWriter

logger = logging.getLogger("e3-routing-capture")

FAMILY_ADAPTERS: Mapping[str, Any] = {
    MoEAdapter.FAMILY: MoEAdapter(),
    MoTAdapter.FAMILY: MoTAdapter(),
    LatentAdapter.FAMILY: LatentAdapter(),
}
SUPPORTED_FAMILIES = frozenset(FAMILY_ADAPTERS)

# Module-path fallback for families that do not set ``_routing_aux_kind``
# (e.g. MoE / MoT). ``moa`` intentionally has no entry and is never recorded.
_PATH_FAMILY_TOKENS = (
    ("latent_mixture", "latent"),
    ("moe", "moe"),
    ("mot", "mot"),
)


def module_family(module: nn.Module) -> str | None:
    """Return the supported routing family for ``module``, or None."""
    kind = getattr(module, "_routing_aux_kind", None)
    if isinstance(kind, str) and kind.lower() in SUPPORTED_FAMILIES:
        return kind.lower()
    module_path = module.__class__.__module__.lower()
    for token, family in _PATH_FAMILY_TOKENS:
        if token in module_path:
            return family
    return None


def discover_routed_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return (named-modules path, module) for every routed module that routed.

    A module qualifies when it satisfies the canonical routed-module protocol
    (``is_routed_module``), maps to a supported family, and produced a non-empty
    ``last_routing_snapshot`` during the latest forward.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    discovered: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not is_routed_module(module):
            continue
        if module_family(module) not in SUPPORTED_FAMILIES:
            continue
        snapshot = getattr(module, "last_routing_snapshot", None)
        if not isinstance(snapshot, dict) or not snapshot:
            continue
        discovered.append((name, module))
    return discovered


def capture_records(
    model: nn.Module,
    *,
    run_id: str,
    captured_at: str,
    forward: Callable[[nn.Module], Any] | None = None,
    step: int | None = None,
    training: bool | None = None,
) -> list[RoutingRecord]:
    """Run ``forward`` once, then adapt every discovered snapshot to a record."""
    if forward is not None:
        forward(model)
    records: list[RoutingRecord] = []
    for name, module in discover_routed_modules(model):
        family = module_family(module)
        if family is None:
            continue  # defensive: discovery already filters unsupported families
        adapter = FAMILY_ADAPTERS[family]
        try:
            record = adapter.to_record(
                module.last_routing_snapshot,
                run_id=run_id,
                captured_at=captured_at,
                module_name=name,
                module_type=module.__class__.__name__,
                step=step,
                training=training,
            )
        except Exception as exc:
            raise RuntimeError(
                f"cannot adapt {family} snapshot for module {name!r} "
                f"({module.__class__.__name__}): {exc}"
            ) from exc
        records.append(record)
    return records


def write_records(records: Sequence[RoutingRecord], output: Path | str) -> int:
    """Persist records with the existing writer (one validated record per line)."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with RoutingRecordWriter(path, append=False) as writer:
        return writer.write_all(records)


def capture_to_file(
    model: nn.Module,
    output: Path | str,
    *,
    run_id: str,
    captured_at: str,
    forward: Callable[[nn.Module], Any] | None = None,
    step: int | None = None,
    training: bool | None = None,
) -> int:
    """Run one forward and write canonical records; returns the line count."""
    records = capture_records(
        model,
        run_id=run_id,
        captured_at=captured_at,
        forward=forward,
        step=step,
        training=training,
    )
    return write_records(records, output)


__all__ = [
    "FAMILY_ADAPTERS",
    "SUPPORTED_FAMILIES",
    "capture_records",
    "capture_to_file",
    "discover_routed_modules",
    "module_family",
    "write_records",
]
