"""Unit tests for the P1-B sample-overhead artifact (builder + verifier)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_sample_capture_overhead import (
    _bootstrap_ci_mean,
    _distribution,
    _percentile_nearest_rank,
    compose_payload,
)
from scripts.run_e3_smoke import verify_sample_overhead_result

_PAIR_COUNT = 120


def _arm_pairs() -> list[tuple[float, float]]:
    """Deterministic sub-second timings around a ~5% ON overhead with noise."""
    pairs: list[tuple[float, float]] = []
    for index in range(_PAIR_COUNT):
        off = 0.2 + 0.001 * (index % 7)
        on = off * 1.05 + 0.0005 * (index % 5)
        pairs.append((off, on))
    return pairs


def _payload(**overrides: object) -> dict:
    payload = compose_payload(
        run_id="run-p1b",
        captured_at="2026-09-09T00:00:00+08:00",
        started_at="2026-09-09T00:00:00+08:00",
        finished_at="2026-09-09T00:00:30+08:00",
        model_config="ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml",
        pairs=_PAIR_COUNT,
        warmup=5,
        size=640,
        arm_pairs=_arm_pairs(),
        environment={"python": "3.11.9", "torch": "2.13.0+cpu", "ultralytics": "8.4.101", "platform": "test"},
    )
    payload.update(overrides)
    return payload


def test_percentile_nearest_rank() -> None:
    ordered = list(range(1, 101))
    assert _percentile_nearest_rank(ordered, 50.0) == 50
    assert _percentile_nearest_rank(ordered, 95.0) == 95


def test_distribution_reports_median_and_p95() -> None:
    stats = _distribution(list(range(1, 101)))
    assert stats["mean"] == pytest.approx(50.5)
    assert stats["median"] == pytest.approx(50.5)
    assert stats["p95"] == 95
    assert stats["min"] == 1 and stats["max"] == 100 and stats["n"] == 100


def test_distribution_rejects_short_or_non_finite_samples() -> None:
    with pytest.raises(ValueError):
        _distribution([1.0])
    with pytest.raises(ValueError):
        _distribution([1.0, float("nan")])


def test_bootstrap_ci_is_deterministic_and_covers_the_mean() -> None:
    values = [float(10 + (index % 5)) for index in range(200)]
    first = _bootstrap_ci_mean(values, resamples=1000, seed=0)
    second = _bootstrap_ci_mean(values, resamples=1000, seed=0)
    assert first == second
    mean = sum(values) / len(values)
    assert first[0] < mean < first[1]
    assert first[0] <= first[1]


def test_compose_payload_records_paired_protocol_metadata() -> None:
    payload = _payload()
    observations = payload["observations"]
    assert len(observations) == _PAIR_COUNT
    assert observations[0]["pair"] == 1
    assert observations[0]["off_ms_per_iteration"] == pytest.approx(200.0)
    assert observations[0]["on_ms_per_iteration"] > observations[0]["off_ms_per_iteration"]
    assert observations[0]["difference_ms"] == pytest.approx(
        observations[0]["on_ms_per_iteration"] - observations[0]["off_ms_per_iteration"]
    )

    protocol = payload["protocol"]
    assert protocol["name"] == "p1-b-sample-overhead"
    assert "alternating paired" in protocol["design"]
    assert protocol["pairing"]["order"] == "OFF then ON within each observation"
    assert protocol["sample_workload"] == {"mot": 4, "moe": 4, "latent": 1}
    assert protocol["threshold"] == "none (statistics only; no pre-registered <10% judgment)"
    assert payload["parameters"]["paired_observations"] == _PAIR_COUNT
    assert payload["parameters"]["bootstrap"]["resamples"] == 10_000
    assert payload["baselines"]["official_base_ref"].startswith("3eb6cd9")
    assert payload["baselines"]["runtime_ultralytics_editable_install_head"] == "d604c4bca8ceba3240c730f1b6e2767b7a320f6c"
    assert payload["baselines"]["baseline_root_head"] == "aa5d2e20c109b96f4a0c68f667ed2694586ef745"
    assert payload["environment"]["ultralytics"] == "8.4.101"


def test_compose_payload_statistics_include_paired_metrics() -> None:
    payload = _payload()
    stats = payload["statistics"]
    overhead = stats["overhead_percent"]
    assert overhead["n"] == _PAIR_COUNT
    for key in ("mean", "std", "median", "p95", "min", "max"):
        assert isinstance(overhead[key], float)
    ci95 = overhead["ci95"]
    assert len(ci95) == 2 and ci95[0] <= overhead["mean"] <= ci95[1]
    assert 3.0 < overhead["mean"] < 8.0  # synthetic pairs are ~5%
    difference = stats["paired_difference_ms"]
    assert difference["mean"] > 0.0
    for key in ("mean", "std", "median", "p95", "min", "max"):
        assert isinstance(difference[key], float)


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


def test_verify_sample_overhead_reports_non_finite_mean(tmp_path: Path) -> None:
    payload = _payload()
    payload["statistics"]["overhead_percent"]["mean"] = float("nan")
    path = tmp_path / "sample_overhead_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = verify_sample_overhead_result(path, run_id="run-p1b")
    assert any("mean missing or non-finite" in error for error in errors)


def test_verify_sample_overhead_reports_inverted_ci95(tmp_path: Path) -> None:
    payload = _payload()
    ci95 = payload["statistics"]["overhead_percent"]["ci95"]
    payload["statistics"]["overhead_percent"]["ci95"] = [ci95[1], ci95[0]]
    path = tmp_path / "sample_overhead_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = verify_sample_overhead_result(path, run_id="run-p1b")
    assert any("ci95" in error for error in errors)


def test_verify_sample_overhead_reports_fewer_than_100_observations(tmp_path: Path) -> None:
    payload = compose_payload(
        run_id="run-p1b",
        captured_at="2026-09-09T00:00:00+08:00",
        started_at="2026-09-09T00:00:00+08:00",
        finished_at="2026-09-09T00:00:20+08:00",
        model_config="ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml",
        pairs=50,
        warmup=5,
        size=640,
        arm_pairs=_arm_pairs()[:50],
        environment={"python": "3.11.9", "torch": "2.13.0+cpu", "ultralytics": "8.4.101", "platform": "test"},
    )
    path = tmp_path / "sample_overhead_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = verify_sample_overhead_result(path, run_id="run-p1b")
    assert any(">= 100" in error for error in errors)


def test_verify_sample_overhead_reports_missing_statistics(tmp_path: Path) -> None:
    payload = _payload()
    del payload["statistics"]
    path = tmp_path / "sample_overhead_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = verify_sample_overhead_result(path, run_id="run-p1b")
    assert any("statistics" in error for error in errors)
