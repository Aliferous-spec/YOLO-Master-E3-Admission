"""Unit tests for the E3 P0 capture wiring (real snapshot -> adapter -> JSONL).

The wiring is exercised with a small ``nn.Module`` tree whose leaf modules mimic
real routed modules (``num_experts`` / ``top_k`` / ``aux_loss`` /
``last_routing_snapshot``).  The unit under test only drives discovery, family
dispatch and JSONL persistence; schema, adapters and the writer are reused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.routing_capture import (
    capture_to_file,
    discover_routed_modules,
    module_family,
)
from scripts.routing_record_writer import iter_records


class _RoutedLeaf(nn.Module):
    """Fake routed module: satisfies the routed-module protocol, family-marked."""

    def __init__(self, *, family: str, num_experts: int = 3, top_k: int = 2) -> None:
        super().__init__()
        self._routing_aux_kind = family
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss = torch.tensor(0.0)
        self.last_routing_snapshot: dict = {}


class _PathRouted(nn.Module):
    """Fake routed module identified only by its (simulated) module path."""

    num_experts = 3
    top_k = 2
    aux_loss = torch.tensor(0.0)

    def __init__(self) -> None:
        super().__init__()
        self.last_routing_snapshot: dict = {}


_PathRouted.__module__ = "ultralytics.nn.modules.moe.unittest"


def _snapshot(num_experts: int = 3, top_k: int = 2, usage: list[float] | None = None) -> dict:
    return {
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_usage": [2.0, 1.0, 1.0] if usage is None else usage,
        "aux_loss": 0.01,
        "finite_diagnostics": {"aux_loss_finite": True},
    }


def _model_with_leaf_modules() -> tuple[nn.Module, dict[str, _RoutedLeaf]]:
    model = nn.Module()
    leaves: dict[str, _RoutedLeaf] = {}
    for name, family in (("moe_0", "moe"), ("mot_0", "mot"), ("lat_0", "latent")):
        leaves[name] = _RoutedLeaf(family=family)
        model.add_module(name, leaves[name])
    return model, leaves


def test_module_family_uses_aux_kind_then_module_path() -> None:
    assert module_family(_RoutedLeaf(family="moe")) == "moe"
    assert module_family(_RoutedLeaf(family="mot")) == "mot"
    assert module_family(_RoutedLeaf(family="latent")) == "latent"
    # Real MoE modules do not carry ``_routing_aux_kind``; the class module path
    # (ultralytics.nn.modules.moe.*) is the family fallback.
    assert module_family(_PathRouted()) == "moe"
    assert module_family(nn.Conv2d(3, 3, 1)) is None


def test_discover_skips_empty_and_unsupported_modules() -> None:
    model = nn.Module()
    moe_leaf = _RoutedLeaf(family="moe")
    model.add_module("moe_0", moe_leaf)
    empty_leaf = _RoutedLeaf(family="moe")
    model.add_module("empty_0", empty_leaf)  # routed but no snapshot this forward
    model.add_module("conv_0", nn.Conv2d(3, 3, 1))

    def forward(root: nn.Module) -> None:
        moe_leaf.last_routing_snapshot = _snapshot()
        # empty_0 is intentionally left with an empty snapshot.

    forward(model)
    discovered = discover_routed_modules(model)
    names = [name for name, _ in discovered]
    assert names == ["moe_0"]
    assert len(discovered) == 1
    assert moe_leaf.last_routing_snapshot  # forward ran and populated the snapshot


def test_capture_writes_one_canonical_record_per_routed_module(tmp_path: Path) -> None:
    model, leaves = _model_with_leaf_modules()
    moa_leaf = _RoutedLeaf(family="moa")
    moa_leaf.num_experts = 4
    moa_leaf.top_k = 2
    model.add_module("moa_0", moa_leaf)  # unsupported family: never recorded

    def forward(root: nn.Module) -> None:
        leaves["moe_0"].last_routing_snapshot = _snapshot(usage=[2.0, 1.0, 1.0])
        leaves["mot_0"].last_routing_snapshot = _snapshot()
        leaves["lat_0"].last_routing_snapshot = _snapshot()
        moa_leaf.last_routing_snapshot = _snapshot(num_experts=4, usage=[2.0, 1.0, 1.0])

    output = tmp_path / "routing_records.jsonl"
    count = capture_to_file(
        model,
        output,
        forward=forward,
        run_id="e3-p0-001",
        captured_at="2026-09-05T10:00:00+08:00",
        step=7,
        training=False,
    )

    assert count == 3
    records = list(iter_records(output))
    assert len(records) == 3
    by_family = {record.family: record for record in records}
    assert set(by_family) == {"moe", "mot", "latent"}

    moe = by_family["moe"]
    assert moe.module.name == "moe_0"
    assert moe.module.type == "_RoutedLeaf"
    assert moe.routing_paradigm == "discrete_selection"
    assert moe.routing.num_experts == 3
    assert moe.routing.top_k == 2
    assert list(moe.routing.expert_usage) == [2.0, 1.0, 1.0]
    assert moe.routing.normalized_expert_load == pytest.approx([0.5, 0.25, 0.25])
    assert moe.step == 7
    assert moe.training is False
    assert moe.run_id == "e3-p0-001"
    assert moe.source_snapshot_keys == tuple(sorted(_snapshot().keys()))

    assert by_family["mot"].routing_paradigm == "scene_conditioned"
    assert by_family["mot"].module.name == "mot_0"
    assert by_family["latent"].routing_paradigm == "continuous_fusion"
    assert by_family["latent"].module.name == "lat_0"

def test_combined_family_captures_write_one_routing_records_jsonl(tmp_path: Path) -> None:
    """P0-1: per-family captures are combined into a single v1 JSONL by main."""
    from scripts.routing_capture import capture_records, write_records

    moe_model = nn.Module()
    moe_leaf = _RoutedLeaf(family="moe")
    moe_model.add_module("moe_a", moe_leaf)

    latent_model = nn.Module()
    latent_leaf = _RoutedLeaf(family="latent")
    latent_model.add_module("lat_a", latent_leaf)

    def moe_forward(root: nn.Module) -> None:
        moe_leaf.last_routing_snapshot = _snapshot()

    def latent_forward(root: nn.Module) -> None:
        latent_leaf.last_routing_snapshot = _snapshot()

    records = capture_records(moe_model, run_id="e3-p0-combined", captured_at="2026-09-05T10:00:00+08:00", forward=moe_forward, training=False)
    records += capture_records(latent_model, run_id="e3-p0-combined", captured_at="2026-09-05T10:00:00+08:00", forward=latent_forward, training=False)

    output = tmp_path / "routing_records.jsonl"
    assert write_records(records, output) == 2
    parsed = list(iter_records(output))
    assert [record.family for record in parsed] == ["moe", "latent"]
    assert {record.module.name for record in parsed} == {"moe_a", "lat_a"}
    assert all(record.run_id == "e3-p0-combined" for record in parsed)
