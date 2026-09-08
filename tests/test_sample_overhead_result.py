"""Unit tests for the P1-B sample-overhead artifact (builder + verifier)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_sample_capture_overhead import (
    _summarize,
    compose_payload,
)
from scripts.run_e3_smoke import verify_sample_overhead_result


def _arm_pairs() -> list[tuple[float, float]]:
    return [(10.0, 12.0), (10.0, 11.0), (10.0, 13.0)]


def _payload(**overrides: object) -> dict:
    payload = compose_payload(
        run_id="run-p1b",
        captured_at="2026-09-08T00:00:00+08:00",
        started_at="2026-09-08T00:00:00+08:00",
        finished_at="2026-09-08T00:00:30+08:00",
        model_config="ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml",
        iterations=50,
        warmup=5,
        repeats=3,
        size=640,
        arm_pairs=_arm_pairs(),
        environment={"python": "3.11.9", "torch": "2.13.0+cpu", "ultralytics": "8.4.101", "platform": "test"},
    )
    payload.update(overrides)
    return payload


def test_summarize_reports_mean_std_min_max_n() -> None:
    stats = _summarize([10.0, 20.0, 30.0])
    assert stats["mean"] == pytest.approx(20.0)
    assert stats["std"] == pytest.approx(10.0)
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(30.0)
    assert stats["n"] == 3


def test_summarize_rejects_short_or_non_finite_samples() -> None:
    with pytest.raises(ValueError):
        _summarize([1.0])
    with pytest.raises(ValueError):
        _summarize([1.0, float("nan")])


def test_compose_payload_records_protocol_metadata() -> None:
    payload = _payload()
    stats = payload["statistics"]["overhead_percent"]
    assert stats["mean"] == pytest.approx(20.0)
    assert stats["std"] == pytest.approx(10.0)
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(30.0)
    assert stats["n"] == 3
    assert len(payload["repetitions"]) == 3
    assert payload["repetitions"][0]["off_ms_per_iteration"] == pytest.approx(200.0)
    assert payload["run_id"] == "run-p1b"
    assert payload["protocol"]["unit"]["overhead"] == "percent"
    assert payload["protocol"]["sample_workload"] == {"mot": 4, "moe": 4, "latent": 1}
    assert "off" in payload["protocol"]["arms"] and "on" in payload["protocol"]["arms"]
    assert payload["protocol"]["threshold"] == "none (statistics only; no pre-registered <10% judgment)"
    assert payload["baselines"]["official_base_ref"].startswith("3eb6cd9")
    assert payload["baselines"]["runtime_ultralytics_editable_install_head"] == "d604c4b"
    assert payload["baselines"]["baseline_root_head"] == "aa5d2e2"
    assert payload["environment"]["ultralytics"] == "8.4.101"


def test_verify_sample_overhead_accepts_valid_artifact(tmp_path: Path) -> None:
    path = tmp_path / "sample_overhead_result.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert verify_sample_overhead_result(path, run_id="run-p1b") == []


def test_verify_sample_overhead_reports_missing_file(tmp_path: Path) -> None:
    errors = verify_sample_overhead_result(tmp_path / "absent.json", run_id="run-p1b")
    assert any("missing" in error for error in errors)


def test_verify_sample_overhead_reports_run_id_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "sample_overhead_result.json"
    path.write_text(json.dumps(_payload(run_id="other-run")), encoding="utf-8")
    errors = verify_sample_overhead_result(path, run_id="run-p1b")
    assert any("run_id mismatch" in error for error in errors)


def test_verify_sample_overhead_reports_non_finite_statistics(tmp_path: Path) -> None:
    payload = _payload()
    payload["statistics"]["overhead_percent"]["mean"] = float("nan")
    path = tmp_path / "sample_overhead_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = verify_sample_overhead_result(path, run_id="run-p1b")
    assert any("non-finite" in error for error in errors)


def test_verify_sample_overhead_reports_fewer_than_three_repeats(tmp_path: Path) -> None:
    payload = compose_payload(
        run_id="run-p1b",
        captured_at="2026-09-08T00:00:00+08:00",
        started_at="2026-09-08T00:00:00+08:00",
        finished_at="2026-09-08T00:00:20+08:00",
        model_config="ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml",
        iterations=50,
        warmup=5,
        repeats=2,
        size=640,
        arm_pairs=[(10.0, 12.0), (10.0, 11.0)],
        environment={"python": "3.11.9", "torch": "2.13.0+cpu", "ultralytics": "8.4.101", "platform": "test"},
    )
    path = tmp_path / "sample_overhead_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = verify_sample_overhead_result(path, run_id="run-p1b")
    assert any(">= 3" in error for error in errors)


def test_verify_sample_overhead_reports_missing_statistics(tmp_path: Path) -> None:
    payload = _payload()
    del payload["statistics"]
    path = tmp_path / "sample_overhead_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = verify_sample_overhead_result(path, run_id="run-p1b")
    assert any("statistics" in error for error in errors)