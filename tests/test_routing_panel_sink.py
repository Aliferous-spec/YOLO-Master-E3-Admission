"""Tests for the P1 routing panel sink.

The panel is an evidence surface, so these tests pin the two properties that
matter for judging: it never invents a family it did not observe, and it never
renders a fabricated zero for data that is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.routing_panel_sink import (
    PANEL_FILENAME,
    REQUIRED_FAMILIES,
    RoutingPanelSink,
    load_records,
    render_panel_html,
    render_run,
)
from scripts.routing_record import RoutingRecord


def _record(
    family: str,
    module_name: str,
    *,
    step: int | None = None,
    load: tuple[float, ...] = (0.5, 0.3, 0.2),
    entropy: float = 0.9,
    gini: float = 0.2,
    dominant_share: float = 0.5,
) -> RoutingRecord:
    total = sum(load)
    normalized = tuple(value / total for value in load) if total else tuple(load)
    return RoutingRecord.from_dict(
        {
            "schema_version": "e3-routing/v1",
            "run_id": "panel-test",
            "captured_at": "2026-09-09T22:00:00+08:00",
            "family": family,
            "routing_paradigm": {
                "moe": "discrete_selection",
                "mot": "scene_conditioned",
                "latent": "continuous_fusion",
            }[family],
            "module": {"name": module_name, "type": "SyntheticRouted"},
            "step": step,
            "training": False,
            "routing": {
                "num_experts": len(load),
                "top_k": 1,
                "expert_usage": list(normalized),
                "normalized_expert_load": list(normalized),
                "routing_entropy_nats": entropy,
                "routing_entropy_normalized": entropy,
                "load_gini": gini,
                "dominant_expert": 0,
                "dominant_expert_share": dominant_share,
            },
        }
    )


def _three_family_records() -> list[RoutingRecord]:
    records: list[RoutingRecord] = []
    for family in REQUIRED_FAMILIES:
        for sample in range(3):
            records.append(
                _record(family, f"model.0.{family}", step=sample, load=(0.4 + 0.1 * sample, 0.3, 0.2))
            )
    return records


# -- coverage ---------------------------------------------------------------


def test_all_three_families_covered(tmp_path: Path) -> None:
    with RoutingPanelSink(tmp_path, run_id="r1") as sink:
        sink.add_records(_three_family_records(), stream="sample")
        meta = sink.close()
    assert meta["families_covered"] == ["latent", "moe", "mot"]
    assert meta["families_missing"] == []
    assert meta["records"]["sample"] == 9
    assert (tmp_path / PANEL_FILENAME).is_file()


def test_missing_family_is_reported_not_fabricated(tmp_path: Path) -> None:
    with RoutingPanelSink(tmp_path, run_id="r2") as sink:
        sink.add_records([_record("moe", "model.0.moe")], stream="canonical")
        meta = sink.close()
    assert meta["families_covered"] == ["moe"]
    # Reported in REQUIRED_FAMILIES order, not sorted: stable across runs.
    assert meta["families_missing"] == ["mot", "latent"]

    page = (tmp_path / PANEL_FILENAME).read_text(encoding="utf-8")
    assert "MISSING" in page
    # The absent families must still appear as sections, explicitly marked.
    assert "Reported as missing, not estimated." in page


def test_missing_metric_is_not_rendered_as_zero(tmp_path: Path) -> None:
    """A metric the schema could not supply renders as n/a, never as 0.0."""
    with RoutingPanelSink(tmp_path, run_id="r3") as sink:
        sink.add_records([_record("moe", "model.0.moe")], stream="canonical")
        # Simulate a family whose adapter could not supply this metric.
        sink.series[("moe", "model.0.moe")].points[0].metrics["load_gini"] = None
        sink.close()
    page = (tmp_path / PANEL_FILENAME).read_text(encoding="utf-8")
    assert "n/a" in page
    assert ">0.0000<" not in page


def test_finite_rejects_non_numeric_and_non_finite() -> None:
    from scripts.routing_panel_sink import _finite

    assert _finite(0.5) == 0.5
    assert _finite(None) is None
    assert _finite("x") is None
    assert _finite(True) is None
    assert _finite(float("nan")) is None
    assert _finite(float("inf")) is None


# -- streams ----------------------------------------------------------------


def test_canonical_and_sample_streams_stay_separate(tmp_path: Path) -> None:
    with RoutingPanelSink(tmp_path, run_id="r4") as sink:
        sink.add_records([_record("moe", "model.0.moe")], stream="canonical")
        sink.add_records([_record("moe", "model.0.moe", step=0)], stream="sample")
        meta = sink.close()
    assert meta["records"] == {"canonical": 1, "sample": 1}


def test_close_is_idempotent(tmp_path: Path) -> None:
    sink = RoutingPanelSink(tmp_path, run_id="r5")
    sink.open()
    sink.add_records([_record("moe", "m")], stream="canonical")
    first = sink.close()
    second = sink.close()
    assert first == second


# -- inputs -----------------------------------------------------------------


def test_accepts_dict_records(tmp_path: Path) -> None:
    with RoutingPanelSink(tmp_path, run_id="r6") as sink:
        sink.add_records([_record("latent", "m").to_dict()], stream="canonical")
        meta = sink.close()
    assert meta["families_covered"] == ["latent"]


def test_rejects_unsupported_record_type(tmp_path: Path) -> None:
    sink = RoutingPanelSink(tmp_path, run_id="r7")
    with pytest.raises(TypeError):
        sink.add_records([object()], stream="canonical")


def test_tensorboard_absent_degrades_to_html_only(tmp_path: Path) -> None:
    """No tensorboard package in the environment -> HTML channel must survive."""
    with RoutingPanelSink(tmp_path, run_id="r8", tensorboard=True) as sink:
        sink.add_records(_three_family_records(), stream="sample")
        meta = sink.close()
    assert "html" in meta["channels"]
    assert (tmp_path / PANEL_FILENAME).is_file()


def test_empty_run_renders_explicit_empty_state(tmp_path: Path) -> None:
    with RoutingPanelSink(tmp_path, run_id="r9") as sink:
        meta = sink.close()
    assert meta["families_covered"] == []
    assert meta["families_missing"] == list(REQUIRED_FAMILIES)
    page = (tmp_path / PANEL_FILENAME).read_text(encoding="utf-8")
    assert "No records captured for this family" in page


# -- round trip -------------------------------------------------------------


def test_render_run_reads_both_jsonl_files(tmp_path: Path) -> None:
    canonical = tmp_path / "routing_records.jsonl"
    sample = tmp_path / "sample_routing_records.jsonl"
    canonical.write_text(
        "\n".join(record.to_json() for record in _three_family_records()), encoding="utf-8"
    )
    sample.write_text(
        "\n".join(
            _record("moe", "model.0.moe", step=index).to_json() for index in range(4)
        ),
        encoding="utf-8",
    )
    meta = render_run(tmp_path, run_id="r10")
    assert meta["families_covered"] == ["latent", "moe", "mot"]
    assert meta["records"] == {"canonical": 9, "sample": 4}
    assert len(load_records(canonical)) == 9


def test_panel_metadata_is_json_serializable(tmp_path: Path) -> None:
    with RoutingPanelSink(tmp_path, run_id="r11") as sink:
        sink.add_records(_three_family_records(), stream="sample")
        meta = sink.close()
    assert json.loads(json.dumps(meta))["run_id"] == "r11"


def test_html_is_self_contained(tmp_path: Path) -> None:
    with RoutingPanelSink(tmp_path, run_id="r12") as sink:
        sink.add_records(_three_family_records(), stream="sample")
        sink.close()
    page = (tmp_path / PANEL_FILENAME).read_text(encoding="utf-8")
    # No external asset references: offline-openable by design.
    assert "http://" not in page
    assert "https://" not in page
    assert "<script" not in page
    assert render_panel_html(sink, {"channels": ["html"]}).startswith("<!DOCTYPE html>")
