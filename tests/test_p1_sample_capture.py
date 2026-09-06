"""Unit tests for E3 P1-A sample-level capture (per-sample module snapshots).

Covers the P1-A acceptance invariants without running real models:

- sample rows never enter the canonical ``routing_records.jsonl`` and the
  P1-A step never exposes a ``records`` count to ``execute_smoke_steps()``;
- per-family ``step`` is continuous 0..S_f-1 and row counts equal S_f x M_f;
- run_id / schema_version stay ``e3-routing/v1`` on every sample row;
- MoE train-mode BN isolation: restoring the initial BN running state before
  every sample forward makes per-sample routing records order-independent.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_e3_smoke as smoke
from scripts.routing_capture import (
    capture_records,
    capture_sample_records,
    restore_bn_running_state,
    snapshot_bn_running_state,
    write_records,
)
from scripts.routing_record_writer import iter_records

CAPTURED_AT = "2026-09-06T00:00:00+08:00"


# ---------------------------------------------------------------------------
# Fakes (mirror real routed modules; adapter/schema/writer are reused)
# ---------------------------------------------------------------------------

class _RoutedLeaf(nn.Module):
    """Fake routed module that refreshes ``last_routing_snapshot`` per input."""

    def __init__(self, family: str, num_experts: int = 3, top_k: int = 2) -> None:
        super().__init__()
        self._routing_aux_kind = family
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss = torch.tensor(0.0)
        self.last_routing_snapshot: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = float(x.mean().item())
        usage = [1.0 + max(value, 0.0), 0.5, 0.5]
        self.last_routing_snapshot = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_usage": usage,
            "aux_loss": 0.01,
            "finite_diagnostics": {"aux_loss_finite": True},
        }
        return x


class _MultiLeafRoot(nn.Module):
    """Host model with N fake routed leaves; ``forward(x)`` refreshes every leaf."""

    def __init__(self, family: str, count: int) -> None:
        super().__init__()
        for index in range(count):
            self.add_module(f"leaf_{index}", _RoutedLeaf(family))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for child in self.children():
            child(x)
        return x


class _BnSensitiveLeaf(nn.Module):
    """Fake routed leaf whose snapshot depends on the BN running_mean."""

    def __init__(self, bn: nn.BatchNorm2d) -> None:
        super().__init__()
        self._bn = bn
        self._routing_aux_kind = "moe"
        self.num_experts = 3
        self.top_k = 2
        self.aux_loss = torch.tensor(0.0)
        self.last_routing_snapshot: dict = {}

    def refresh(self) -> None:
        mean = float(self._bn.running_mean[0].item())
        usage = [1.0 + mean, 1.0 - mean, 1.0]
        self.last_routing_snapshot = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_usage": usage,
            "aux_loss": 0.01,
            "finite_diagnostics": {"aux_loss_finite": True},
        }


class _BnSampleModel(nn.Module):
    """Train-mode host: a real BatchNorm updates running stats per forward."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3, momentum=0.9)
        self.leaf = _BnSensitiveLeaf(self.bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(x)
        self.leaf.refresh()
        return y


def _snapshot_inputs(count: int) -> list[torch.Tensor]:
    return [torch.full((1, 3, 8, 8), float(value)) for value in range(count)]


def _canonical_records(family: str, run_id: str) -> list:
    model = _MultiLeafRoot(family, 1)
    return capture_records(
        model,
        run_id=run_id,
        captured_at=CAPTURED_AT,
        forward=lambda root: root(torch.zeros(1, 3, 8, 8)),
        step=None,
        training=False,
    )


def _sample_records(family: str, *, samples: int, modules: int, run_id: str) -> list:
    model = _MultiLeafRoot(family, modules)
    return capture_sample_records(
        model,
        _snapshot_inputs(samples),
        run_id=run_id,
        captured_at=CAPTURED_AT,
        training=False,
    )


# ---------------------------------------------------------------------------
# 1. Canonical routing_records.jsonl never contains sample rows
# ---------------------------------------------------------------------------

def test_p1_sample_step_keeps_canonical_jsonl_clean(tmp_path: Path, monkeypatch) -> None:
    run_id = "p1-run-001"
    artifacts = tmp_path / "artifacts" / "smoke" / run_id
    artifacts.mkdir(parents=True)
    logger = logging.getLogger("test-p1-canonical")

    monkeypatch.setattr(smoke, "_capture_mot_sample_records", lambda cfg, root, rid: _sample_records("mot", samples=3, modules=2, run_id=rid))
    monkeypatch.setattr(smoke, "_capture_moe_sample_records", lambda cfg, rid: _sample_records("moe", samples=4, modules=1, run_id=rid))
    monkeypatch.setattr(smoke, "_capture_latent_sample_records", lambda cfg, rid: _sample_records("latent", samples=1, modules=3, run_id=rid))

    config: dict = {"mot": {}, "moe": {}, "latent": {}}
    canonical = [_canonical_records("mot", run_id), _canonical_records("moe", run_id), _canonical_records("latent", run_id)]
    steps = {
        "mot": lambda: {"records": canonical[0]},
        "moe": lambda: {"records": canonical[1]},
        "latent": lambda: {"records": canonical[2]},
        "overhead": lambda: {"returncode": 0},
        "samples": lambda: smoke.run_sample_capture(config, artifacts, logger, run_id=run_id, baseline_root=tmp_path),
    }
    summary, collected = smoke.execute_smoke_steps(steps, logger)

    assert summary["status"] == "PASS"
    assert summary["steps"]["samples"]["status"] == "PASS"
    assert "records" not in summary["steps"]["samples"]  # acceptance: no records count

    canonical_path = artifacts / "routing_records.jsonl"
    written = write_records(collected, canonical_path)
    assert written == len(canonical[0]) + len(canonical[1]) + len(canonical[2])
    assert len(collected) == 3

    canonical_rows = list(iter_records(canonical_path))
    assert canonical_rows  # P0 rows are still produced
    assert all(row.step is None for row in canonical_rows)  # no sample rows in canonical

    sample_path = artifacts / "sample_routing_records.jsonl"
    assert sample_path.is_file()
    assert smoke.verify_sample_records(sample_path, run_id=run_id) == []
    sample_rows = list(iter_records(sample_path))
    assert all(row.step is not None for row in sample_rows)
    # Primary keys (run_id, family, module.name, step) never overlap.
    canonical_keys = {(r.run_id, r.family, r.module.name, r.step) for r in canonical_rows}
    sample_keys = {(r.run_id, r.family, r.module.name, r.step) for r in sample_rows}
    assert not (canonical_keys & sample_keys)


# ---------------------------------------------------------------------------
# 2/3. step continuity and row counts read back from the sample file
# ---------------------------------------------------------------------------

def test_sample_file_rows_are_continuous_and_count_correct(tmp_path: Path, monkeypatch) -> None:
    run_id = "p1-run-002"
    artifacts = tmp_path / "smoke" / run_id
    artifacts.mkdir(parents=True)
    logger = logging.getLogger("test-p1-counts")

    monkeypatch.setattr(smoke, "_capture_mot_sample_records", lambda cfg, root, rid: _sample_records("mot", samples=3, modules=2, run_id=rid))
    monkeypatch.setattr(smoke, "_capture_moe_sample_records", lambda cfg, rid: _sample_records("moe", samples=4, modules=1, run_id=rid))
    monkeypatch.setattr(smoke, "_capture_latent_sample_records", lambda cfg, rid: _sample_records("latent", samples=1, modules=3, run_id=rid))

    result = smoke.run_sample_capture({"mot": {}, "moe": {}, "latent": {}}, artifacts, logger, run_id=run_id, baseline_root=tmp_path)
    assert result["sample_file"] == "sample_routing_records.jsonl"
    assert "records" not in result

    rows = list(iter_records(artifacts / "sample_routing_records.jsonl"))
    assert len(rows) == 3 * 2 + 4 * 1 + 1 * 3

    expected = {"mot": (3, 2), "moe": (4, 1), "latent": (1, 3)}
    by_family: dict[str, list] = defaultdict(list)
    for row in rows:
        by_family[row.family].append(row)
    for family, (sample_count, module_count) in expected.items():
        family_rows = by_family[family]
        assert len(family_rows) == sample_count * module_count  # S_f x M_f
        assert sorted({row.step for row in family_rows}) == list(range(sample_count))  # continuous
        assert len({row.module.name for row in family_rows}) == module_count
        keys = [(row.step, row.module.name) for row in family_rows]
        assert len(keys) == len(set(keys))  # primary key uniqueness


def test_sample_rows_run_id_and_schema(tmp_path: Path, monkeypatch) -> None:
    run_id = "p1-run-003"
    artifacts = tmp_path / "smoke" / run_id
    artifacts.mkdir(parents=True)
    logger = logging.getLogger("test-p1-schema")

    monkeypatch.setattr(smoke, "_capture_mot_sample_records", lambda cfg, root, rid: _sample_records("mot", samples=1, modules=1, run_id=rid))
    monkeypatch.setattr(smoke, "_capture_moe_sample_records", lambda cfg, rid: _sample_records("moe", samples=1, modules=1, run_id=rid))
    monkeypatch.setattr(smoke, "_capture_latent_sample_records", lambda cfg, rid: _sample_records("latent", samples=1, modules=1, run_id=rid))

    smoke.run_sample_capture({"mot": {}, "moe": {}, "latent": {}}, artifacts, logger, run_id=run_id, baseline_root=tmp_path)
    rows = list(iter_records(artifacts / "sample_routing_records.jsonl"))
    assert rows
    for row in rows:
        assert row.run_id == run_id
        assert row.schema_version == "e3-routing/v1"
        assert isinstance(row.step, int) and row.step >= 0


# ---------------------------------------------------------------------------
# 4. capture_sample_records continuity at the helper level
# ---------------------------------------------------------------------------

def test_capture_sample_records_steps_are_continuous_per_sample() -> None:
    model = _MultiLeafRoot("latent", 2)
    inputs = _snapshot_inputs(4)
    records = capture_sample_records(model, inputs, run_id="p1-run-777", captured_at=CAPTURED_AT, training=False)

    assert len(records) == 4 * 2  # S_f x M_f
    assert sorted({record.step for record in records}) == [0, 1, 2, 3]
    assert len({record.module.name for record in records}) == 2
    keys = [(record.step, record.module.name) for record in records]
    assert len(keys) == len(set(keys))
    for record in records:
        assert isinstance(record.step, int) and record.step >= 0
        assert record.run_id == "p1-run-777"
        assert record.schema_version == "e3-routing/v1"
        assert record.training is False


# ---------------------------------------------------------------------------
# 5. MoE train-mode BN isolation (order invariance)
# ---------------------------------------------------------------------------

def _run_bn_samples(tags: list[float], *, restore: bool, run_id: str, calls: list | None = None) -> dict[float, tuple[float, ...]]:
    model = _BnSampleModel().train()
    samples = [torch.full((1, 3, 8, 8), tag) for tag in tags]
    bn_snapshot = snapshot_bn_running_state(model)

    def prepare(root: nn.Module, step: int) -> None:
        if calls is not None:
            calls.append(step)
        if restore:
            restore_bn_running_state(root, bn_snapshot)

    records = capture_sample_records(
        model, samples, run_id=run_id, captured_at=CAPTURED_AT, training=True, before_each=prepare
    )
    by_tag: dict[float, tuple[float, ...]] = {}
    for record in records:
        usage = tuple(round(float(value), 6) for value in record.routing.expert_usage)
        by_tag[tags[record.step]] = usage
    return by_tag


def test_moe_sample_order_is_invariant_with_bn_restore() -> None:
    order_a = [-0.6, 0.4, 0.7]
    order_b = [0.7, -0.6, 0.4]
    order_c = [-0.6, 0.7, -0.6, 0.4]  # repeated sample included

    out_a = _run_bn_samples(order_a, restore=True, run_id="p1-bn-a")
    out_b = _run_bn_samples(order_b, restore=True, run_id="p1-bn-b")
    out_c = _run_bn_samples(order_c, restore=True, run_id="p1-bn-c")

    for tag in order_a:
        assert out_a[tag] == out_b[tag]
    for tag in order_c:
        assert out_c[tag] == out_a[tag]  # duplicates keep the same routing value
    # Same input at different step positions yields the same record.
    assert out_b[0.7] == out_a[0.7]
    assert out_b[-0.6] == out_a[-0.6]


def test_moe_sample_order_drifts_without_bn_restore() -> None:
    # Negative control: without restoring BN, the same input routed at a later
    # position sees accumulated running statistics and produces a different row.
    out_first = _run_bn_samples([-0.6, 0.4], restore=False, run_id="p1-bn-drift-a")
    out_second = _run_bn_samples([0.4, -0.6], restore=False, run_id="p1-bn-drift-b")
    assert out_first[-0.6] != out_second[-0.6]


def test_bn_restore_runs_before_every_sample() -> None:
    calls: list[int] = []
    _run_bn_samples([0.1, 0.2, 0.3], restore=True, run_id="p1-bn-calls", calls=calls)
    assert calls == [0, 1, 2]


# ---------------------------------------------------------------------------
# BN snapshot / restore helpers
# ---------------------------------------------------------------------------

def test_bn_snapshot_restore_roundtrips_exactly() -> None:
    model = nn.Sequential()
    bn = nn.BatchNorm2d(3, momentum=0.9)
    bn.running_mean.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
    bn.running_var.data.copy_(torch.tensor([4.0, 5.0, 6.0]))
    bn.num_batches_tracked.data.copy_(torch.tensor(7))
    model.add_module("bn", bn)
    model.add_module("untracked", nn.BatchNorm2d(3, track_running_stats=False))

    snapshot = snapshot_bn_running_state(model)
    assert set(snapshot) == {"bn"}  # only track_running_stats BN is recorded

    model.train()
    model(torch.randn(2, 3, 8, 8))
    assert not torch.equal(model.bn.running_mean, snapshot["bn"]["running_mean"])

    restore_bn_running_state(model, snapshot)
    assert torch.equal(model.bn.running_mean, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(model.bn.running_var, torch.tensor([4.0, 5.0, 6.0]))
    assert int(model.bn.num_batches_tracked.item()) == 7


def test_bn_restore_rejects_stale_or_untracked_modules() -> None:
    source = nn.Sequential(nn.BatchNorm2d(3))
    snapshot = snapshot_bn_running_state(source)

    # A module name that does not exist on the target model.
    with pytest.raises(ValueError, match="no longer exists"):
        restore_bn_running_state(nn.BatchNorm2d(3), snapshot)

    # A target module that is no longer a track_running_stats BatchNorm.
    host = nn.Sequential(nn.BatchNorm2d(3, track_running_stats=False))
    with pytest.raises(ValueError, match="no longer a track_running_stats BatchNorm"):
        restore_bn_running_state(host, snapshot)

    with pytest.raises(TypeError, match="BN snapshot must be a mapping"):
        restore_bn_running_state(nn.BatchNorm2d(3), [])

# ---------------------------------------------------------------------------
# verify_sample_records (spec section 8 checks on the sample file)
# ---------------------------------------------------------------------------

def test_verify_sample_records_reports_run_id_and_duplicates(tmp_path: Path) -> None:
    records = _sample_records("moe", samples=3, modules=2, run_id="v-run")
    sample_path = tmp_path / "sample_routing_records.jsonl"
    write_records(records, sample_path)

    assert smoke.verify_sample_records(sample_path, run_id="v-run") == []
    errors = smoke.verify_sample_records(sample_path, run_id="other-run")
    assert any("run_id mismatch" in error for error in errors)

    dup_path = tmp_path / "duplicate.jsonl"
    write_records(records + records[:1], dup_path)
    dup_errors = smoke.verify_sample_records(dup_path, run_id="v-run")
    assert any("duplicate sample row key" in error for error in dup_errors)


def test_verify_sample_records_reports_gap_steps_and_non_int_steps(tmp_path: Path) -> None:
    records = _sample_records("latent", samples=4, modules=1, run_id="v-run")

    gap_path = tmp_path / "gap.jsonl"
    write_records([record for record in records if record.step != 1], gap_path)
    gap_errors = smoke.verify_sample_records(gap_path, run_id="v-run")
    assert any("not continuous" in error for error in gap_errors)

    # A canonical (step=None) row inside the sample file must be rejected.
    bad_path = tmp_path / "step-none.jsonl"
    write_records([_canonical_records("moe", "v-run")[0]], bad_path)
    bad_errors = smoke.verify_sample_records(bad_path, run_id="v-run")
    assert any("int step" in error for error in bad_errors)


def test_verify_sample_records_reports_missing_and_invalid_files(tmp_path: Path) -> None:
    missing = smoke.verify_sample_records(tmp_path / "nope.jsonl", run_id="v-run")
    assert any("missing" in error for error in missing)

    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text("{not valid json}\n", encoding="utf-8")
    invalid = smoke.verify_sample_records(invalid_path, run_id="v-run")
    assert any("invalid" in error for error in invalid)

