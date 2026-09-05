"""Unit tests for E3 P0 run_e3_smoke behaviours.

Covers:
- P0-2 failure isolation across family steps and summary timing/status/error.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_e3_smoke import (
    execute_smoke_steps,
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

