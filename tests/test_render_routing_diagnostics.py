"""Tests for the demo diagnostic renderer (no model, no evidence run needed).

The two figures are the demo's honesty surface: their captions must be counted
from the JSON they point at, and the script must find its defaults relative to
the repo instead of the caller's working directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.render_routing_diagnostics import (
    HEALTH_FILENAME,
    REPO_ROOT,
    TEMPERATURE_FILENAME,
    main,
    parse_args,
    render_diagnostics,
    resolve_path,
    summarize_families,
    summarize_temperature,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _health_module(family: str, layer: str, gini: float, collapse: bool, uniform: bool) -> dict:
    return {
        "family": family,
        "layer": layer,
        "gini": gini,
        "collapse": collapse,
        "uniform": uniform,
    }


def _write_health(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "modules": [
                    _health_module("moe", "model.5", 0.880162, True, False),
                    _health_module("moe", "model.8", 0.971927, True, False),
                    _health_module("latent", "model.23", 0.0, False, True),
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _layer(moved: bool, dominant_changed: bool = False) -> dict:
    delta = {"top1_share": -0.020623, "entropy_normalized": 0.000923, "gini": -0.002578}
    if not moved:
        delta = {key: 0.0 for key in delta}
    return {
        "baseline": {"top1_share": 0.541292, "entropy_normalized": 0.248769, "gini": 0.880162},
        "factor_2_0": {"top1_share": 0.520669, "entropy_normalized": 0.249692, "gini": 0.877584},
        "delta": delta,
        "dominant_expert_changed": dominant_changed,
    }


def _write_temperature(path: Path) -> Path:
    path.write_text(
        json.dumps({"layers": {"model.5": _layer(moved=True), "model.8": _layer(moved=False)}}),
        encoding="utf-8",
    )
    return path


def test_parse_args_resolves_defaults_against_repo_root() -> None:
    args = parse_args([])

    assert args.health_json == REPO_ROOT / (
        "artifacts/health/smoke-20260905-204546-6c7389/routing_health.json"
    )
    assert args.temperature_json.is_absolute()
    assert args.out_dir == REPO_ROOT / "artifacts/demo"


def test_parse_args_keeps_absolute_paths_and_joins_relative_ones(tmp_path: Path) -> None:
    assert resolve_path(tmp_path / "x.json") == tmp_path / "x.json"
    assert resolve_path("build/demo") == REPO_ROOT / "build" / "demo"

    args = parse_args(
        [
            "--health-json", "in/health.json",
            "--temperature-json", "in/temp.json",
            "--out-dir", "out",
        ]
    )

    assert args.health_json == REPO_ROOT / "in" / "health.json"
    assert args.temperature_json == REPO_ROOT / "in" / "temp.json"
    assert args.out_dir == REPO_ROOT / "out"


def test_summaries_are_counted_from_the_records() -> None:
    modules = [
        _health_module("moe", "model.5", 0.88, True, False),
        _health_module("moe", "model.8", 0.97, True, False),
        _health_module("latent", "model.23", 0.0, False, True),
    ]
    assert summarize_families(modules) == "MoE 2/2 collapse  ·  Latent 1/1 uniform"

    assert summarize_temperature({"model.5": _layer(True)}) == (
        "factor = 2.0  ·  1/1 layers changed  ·  dominant expert unchanged"
    )
    assert summarize_temperature({"model.5": _layer(False)}) == (
        "factor = 2.0  ·  0/1 layers changed  ·  dominant expert unchanged"
    )
    assert "dominant expert changed" in summarize_temperature(
        {"model.5": _layer(True, dominant_changed=True)}
    )


def test_render_writes_both_figures(tmp_path: Path) -> None:
    health = _write_health(tmp_path / "routing_health.json")
    temperature = _write_temperature(tmp_path / "moe_temperature_probe.json")
    out_dir = tmp_path / "out"

    written = render_diagnostics(health, temperature, out_dir)

    assert [path.name for path in written] == [HEALTH_FILENAME, TEMPERATURE_FILENAME]
    for path in written:
        assert path.read_bytes()[:8] == PNG_MAGIC


def test_main_runs_from_cli_paths(tmp_path: Path) -> None:
    health = _write_health(tmp_path / "routing_health.json")
    temperature = _write_temperature(tmp_path / "moe_temperature_probe.json")
    out_dir = tmp_path / "cli-out"

    assert main(
        [
            "--health-json", str(health),
            "--temperature-json", str(temperature),
            "--out-dir", str(out_dir),
        ]
    ) == 0
    assert (out_dir / HEALTH_FILENAME).is_file()
    assert (out_dir / TEMPERATURE_FILENAME).is_file()