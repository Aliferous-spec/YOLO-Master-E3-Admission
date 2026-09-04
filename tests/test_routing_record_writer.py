"""Unit tests for the RoutingRecord v1 JSONL writer (E3 P0)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.routing_record import SCHEMA_VERSION, RoutingRecord  # noqa: E402
from scripts.routing_record_writer import (  # noqa: E402
    RoutingRecordWriter,
    RoutingRecordWriterError,
    iter_records,
)

_FAMILY_PARADIGM = {
    "moe": "discrete_selection",
    "mot": "scene_conditioned",
    "latent": "continuous_fusion",
}


def _record(
    *,
    run_id: str = "e3-writer-001",
    step: int | None = 1,
    training: bool | None = True,
    family: str = "moe",
    family_data: dict | None = None,
    source_snapshot_keys: list | None = None,
) -> RoutingRecord:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "captured_at": "2026-09-03T10:00:00+08:00",
        "family": family,
        "routing_paradigm": _FAMILY_PARADIGM[family],
        "module": {"name": "model.5.routing", "type": "TopKMoERouter"},
        "step": step,
        "training": training,
        "routing": {
            "num_experts": 4,
            "top_k": 2,
            "expert_usage": [3.0, 1.0, 0.0, 0.0],
            "normalized_expert_load": [0.75, 0.25, 0.0, 0.0],
            "routing_entropy_nats": 0.5623351446188083,
            "routing_entropy_normalized": 0.4056390622295663,
            "load_gini": 0.625,
            "dominant_expert": 0,
            "dominant_expert_share": 0.75,
            "mean_mixing_weights": [0.8, 0.2, 0.0, 0.0],
        },
        "aux_loss": {"value": 0.0123, "finite": True},
        "usage_scope": "rank_local",
        "global_usage_available": False,
        "family_data": {} if family_data is None else family_data,
        "source_snapshot_keys": (
            ["num_experts", "top_k", "expert_usage"]
            if source_snapshot_keys is None
            else source_snapshot_keys
        ),
    }
    return RoutingRecord.from_dict(payload)


def test_single_record_writes_one_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    record = _record(run_id="e3-writer-single")
    with RoutingRecordWriter(path) as writer:
        assert writer.path == path
        assert writer.closed is False
        writer.write(record)
    assert writer.closed is True

    text = path.read_text(encoding="utf-8")
    assert text == record.to_json() + "\n"
    assert "\r" not in text
    assert len(text.splitlines()) == 1


def test_append_multiple_records_across_sessions(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    first = _record(run_id="e3-writer-a", step=1)
    second = _record(run_id="e3-writer-b", step=2)
    third = _record(run_id="e3-writer-c", step=3, family="latent")

    with RoutingRecordWriter(path) as writer:
        writer.write(first)
        writer.write(second)
    with RoutingRecordWriter(path, append=True) as writer:
        count = writer.write_all([third])
    assert count == 1

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [first.to_json(), second.to_json(), third.to_json()]


def test_each_jsonl_line_parses_independently(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [_record(run_id=f"e3-writer-line-{index}") for index in range(3)]
    with RoutingRecordWriter(path) as writer:
        writer.write_all(records)

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == len(records)
    for index, (raw, original) in enumerate(zip(raw_lines, records), start=1):
        payload = json.loads(raw)
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["run_id"] == f"e3-writer-line-{index - 1}"
        assert RoutingRecord.from_json(raw) == original
        assert json.loads(original.to_json()) == payload


def test_new_file_truncate_and_append_defaults(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    record = _record(run_id="e3-writer-truncate")

    # append=False creates a new file and truncates any existing content.
    with RoutingRecordWriter(path, append=False) as writer:
        writer.write(record)
    assert path.read_text(encoding="utf-8") == record.to_json() + "\n"

    # Default mode is append; reopening appends instead of truncating.
    second = _record(run_id="e3-writer-second", step=2)
    with RoutingRecordWriter(path) as writer:
        writer.write(second)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [record.to_json(), second.to_json()]

    # append=False truncates an existing file back to a single record.
    third = _record(run_id="e3-writer-third", step=3)
    with RoutingRecordWriter(path, append=False) as writer:
        writer.write(third)
    assert path.read_text(encoding="utf-8") == third.to_json() + "\n"


def test_empty_and_new_file_handling(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with pytest.raises(FileNotFoundError):
        list(iter_records(path))  # missing files fail loudly, never silently

    path.write_text("", encoding="utf-8")
    assert list(iter_records(path)) == []

    record = _record(run_id="e3-writer-empty")
    with RoutingRecordWriter(path) as writer:
        writer.write(record)
    assert list(iter_records(path)) == [record]

    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n\n", encoding="utf-8")
    assert list(iter_records(blank)) == []


def test_illegal_and_non_serializable_inputs_rejected(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"

    # Raw upstream snapshot dicts must never reach the JSONL file.
    with RoutingRecordWriter(path) as writer:
        with pytest.raises(TypeError, match="RoutingRecord"):
            writer.write({"expert_usage": [1.0, 1.0], "num_experts": 2})
        with pytest.raises(TypeError, match="RoutingRecord"):
            writer.write(None)
    assert path.read_text(encoding="utf-8") == ""

    # A RoutingRecord carrying non-JSON-serializable family_data fails loudly.
    bad_record = _record(run_id="e3-writer-bad", family_data={"opaque": {1, 2, 3}})
    with RoutingRecordWriter(path) as writer:
        with pytest.raises(RoutingRecordWriterError, match="e3-writer-bad"):
            writer.write(bad_record)
    assert path.read_text(encoding="utf-8") == ""


def test_roundtrip_restores_records_and_structure(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        _record(
            run_id="e3-writer-moe",
            family="moe",
            family_data={"topk_counts": [3.0, 2.0, 1.0, 0.0]},
            source_snapshot_keys=["num_experts", "top_k", "expert_usage", "mean_router_probs"],
        ),
        _record(
            run_id="e3-writer-mot",
            family="mot",
            family_data={"scene_aware": True, "dispatch": {"mode": "dense"}},
            source_snapshot_keys=["num_experts", "top_k", "expert_usage", "scene_aware"],
        ),
        _record(
            run_id="e3-writer-latent",
            family="latent",
            step=None,
            training=None,
            family_data={"value_fusion_mode": "weighted_sum"},
            source_snapshot_keys=["num_experts", "top_k", "expert_usage", "finite"],
        ),
    ]
    with RoutingRecordWriter(path) as writer:
        assert writer.write_all(records) == 3

    restored = list(iter_records(path))
    assert restored == records
    for original, rebuilt in zip(records, restored):
        assert rebuilt.schema_version == SCHEMA_VERSION
        assert rebuilt.source_snapshot_keys == original.source_snapshot_keys
        assert rebuilt.family_data == original.family_data
        assert rebuilt.family == original.family
        assert rebuilt.step == original.step
        assert rebuilt.training == original.training


def test_corrupt_line_reports_path_and_line_number(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    record = _record(run_id="e3-writer-corrupt")
    with RoutingRecordWriter(path) as writer:
        writer.write(record)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"schema_version": "not-json"\n')
    with pytest.raises(ValueError, match="records.jsonl:2"):
        list(iter_records(path))