"""Unit tests for E3 P0 run_e3_smoke behaviours.

Covers:
- P0-2 failure isolation across family steps and summary timing/status/error.
- P0-3 unique auto run_id, --run-id override, per-run artifact isolation.
- P0-4 overhead percent parsing (signed values, NaN/Inf rejection) and
      idempotent ``setup_logging``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_e3_smoke import (
    build_manifest,
    execute_smoke_steps,
    generate_run_id,
    parse_overhead_percent,
    resolve_run_id,
    setup_logging,
    smoke_artifacts_dir,
)

# ---------------------------------------------------------------------------
# P0-2: failure isolation
# ---------------------------------------------------------------------------

def _ok_step(name: str, ran: list[str]):
    def step() -> dict:
        ran.append(name)
        return {"records": []}

    return step

def _boom_step(name: str, ran: list[str], message: str = "exploded"):
    def step() -> dict:
        ran.append(name)
        raise RuntimeError(message)

    return step

def test_execute_smoke_steps_does_not_block_later_steps() -> None:
    ran: list[str] = []
    logger = logging.getLogger("test-e3-isolation")
    summary, records = execute_smoke_steps(
        {
            "mot": _ok_step("mot", ran),
            "moe": _boom_step("moe", ran),
            "latent": _ok_step("latent", ran),
            "overhead": _boom_step("overhead", ran, "overhead is broken"),
        },
        logger,
    )

    assert ran == ["mot", "moe", "latent", "overhead"]  # later families still ran
    assert summary["status"] == "FAIL"
    assert summary["steps"]["mot"]["status"] == "PASS"
    assert summary["steps"]["moe"]["status"] == "FAIL"
    assert "exploded" in summary["steps"]["moe"]["error"]
    assert summary["steps"]["latent"]["status"] == "PASS"
    assert summary["steps"]["overhead"]["status"] == "FAIL"
    assert "overhead is broken" in summary["error"]
    assert records == []

def test_execute_smoke_steps_records_times() -> None:
    logger = logging.getLogger("test-e3-times")
    summary, _ = execute_smoke_steps({"a": _ok_step("a", [])}, logger)
    assert summary["status"] == "PASS"
    assert summary["started_at"]
    assert summary["finished_at"]
    assert summary["finished_at"] >= summary["started_at"]
    step = summary["steps"]["a"]
    assert step["started_at"]
    assert step["finished_at"]
    assert step["status"] == "PASS"
    assert "error" not in step

# ---------------------------------------------------------------------------
# P0-3: run_id and artifact isolation
# ---------------------------------------------------------------------------

def test_generate_run_id_is_unique() -> None:
    assert generate_run_id() != generate_run_id()

def test_resolve_run_id_prefers_cli_and_auto_generates_unique_default() -> None:
    assert resolve_run_id({}, "my-run-1") == "my-run-1"
    first = resolve_run_id({"run_id": "admission-20260825"}, None)
    second = resolve_run_id({"run_id": "admission-20260825"}, None)
    assert first != second  # unique per run even when config pins a legacy id
    assert first != "admission-20260825"

def test_smoke_artifacts_dir_is_per_run(tmp_path: Path) -> None:
    first = smoke_artifacts_dir(tmp_path, "run-one")
    second = smoke_artifacts_dir(tmp_path, "run-two")
    assert first != second
    assert first.parent == tmp_path / "artifacts" / "smoke"
    assert first.name == "run-one"
    assert second.name == "run-two"

def test_two_runs_keep_manifests_isolated(tmp_path: Path) -> None:
    first = smoke_artifacts_dir(tmp_path, "run-a")
    second = smoke_artifacts_dir(tmp_path, "run-b")
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "evidence-a.txt").write_text("a", encoding="utf-8")
    (second / "evidence-b.txt").write_text("b", encoding="utf-8")

    manifest_a = build_manifest(first)
    manifest_b = build_manifest(second)
    assert set(manifest_a) == {"evidence-a.txt"}
    assert set(manifest_b) == {"evidence-b.txt"}
    assert not (set(manifest_a) & set(manifest_b))

# ---------------------------------------------------------------------------
# P0-4: overhead parsing and logging idempotency
# ---------------------------------------------------------------------------

def test_parse_overhead_percent_accepts_signed_values() -> None:
    assert parse_overhead_percent("overhead: -5.01%") == pytest.approx(-5.01)
    assert parse_overhead_percent("overhead: +1.20%") == pytest.approx(1.2)
    assert parse_overhead_percent("overhead: 0.99%") == pytest.approx(0.99)
    assert parse_overhead_percent("overhead: 1.50% (target < 10%)") == pytest.approx(1.5)

def test_parse_overhead_percent_rejects_non_finite_and_missing() -> None:
    for text in ("overhead: nan%", "overhead: inf%", "overhead: -inf%", "no overhead", ""):
        with pytest.raises(ValueError):
            parse_overhead_percent(text)

def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path)
    before = list(logger.handlers)
    assert before
    again = setup_logging(tmp_path)
    assert again is logger
    assert list(logger.handlers) == before  # no duplicate handlers

def test_setup_logging_replaces_file_handler_for_new_run(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "run-a")
    setup_logging(tmp_path / "run-b")
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename) == (tmp_path / "run-b" / "full.log").resolve()

