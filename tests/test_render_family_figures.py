"""Tests for the P0 static family-figure renderer.

These figures are evidence surfaces: each one must be regenerable from the run
directory alone, must carry a traceable source hash, and must never touch the
upstream-owned MoT figure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.render_family_figures import (
    FIGURE_FILENAMES,
    load_moe_matrix,
    render_family_figures,
)
from scripts.routing_record import RoutingRecord

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _latent_line(run_id: str, module: str) -> str:
    usage = [0.25, 0.25, 0.25, 0.25]
    return RoutingRecord.from_dict(
        {
            "schema_version": "e3-routing/v1",
            "run_id": run_id,
            "captured_at": "2026-09-05T20:46:04+08:00",
            "family": "latent",
            "routing_paradigm": "continuous_fusion",
            "module": {"name": module, "type": "LatentMixture"},
            "routing": {
                "num_experts": 4,
                "top_k": 4,
                "expert_usage": usage,
                "normalized_expert_load": usage,
                "routing_entropy_nats": 1.3862943649291992,
                "routing_entropy_normalized": 1.0,
                "load_gini": 0.0,
                "dominant_expert": 0,
                "dominant_expert_share": 0.25,
            },
            "family_data": {"dispatch_policy": "dense", "routing_axis": "expert"},
        }
    ).to_json()


def _write_run(root: Path, run_id: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (run_dir / "moe_usage_stats.json").write_text(
        json.dumps(
            {
                "model.5.routing": {"0": {"hits": 1.0}, "1": {"hits": 3.0}},
                "model.8.routing": {"0": {"hits": 2.0}, "2": {"hits": 2.0}},
            }
        ),
        encoding="utf-8",
    )
    lines = [_latent_line(run_id, "model.23"), _latent_line(run_id, "model.24")]
    (run_dir / "routing_records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def test_moe_matrix_normalizes_hits_per_layer(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "fig-run")
    rows, cols, matrix = load_moe_matrix(run_dir)
    assert rows == ["model.5.routing", "model.8.routing"]
    assert cols == [0, 1, 2]
    assert matrix == [[0.25, 0.75, 0.0], [0.5, 0.0, 0.5]]


def test_render_writes_moe_and_latent_figures_traceable_to_source(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "fig-run")
    out_dir = tmp_path / "figures"

    metadata = render_family_figures(run_dir, out_dir)

    assert metadata["run_id"] == "fig-run"
    assert [figure["family"] for figure in metadata["figures"]] == ["moe", "latent"]
    for family, figure in zip(("moe", "latent"), metadata["figures"]):
        png = out_dir / FIGURE_FILENAMES[family]
        assert png.read_bytes()[:8] == PNG_MAGIC
        assert figure["source_sha256"] == _sha256(run_dir / Path(figure["source"]).name)
    assert (out_dir / "figures.json").is_file()
    assert not list(out_dir.glob("mot*"))


def test_render_rejects_mot(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "fig-run")
    with pytest.raises(ValueError):
        render_family_figures(run_dir, tmp_path / "figures", families=("mot",))