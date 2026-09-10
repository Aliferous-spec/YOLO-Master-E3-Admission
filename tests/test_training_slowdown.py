"""Unit tests for the E3 training-slowdown measurement (protocol + artifact schema).

Pure-logic tests only: the ABBA block layout, paired-observation math, artifact
composition and path validation never import torch/ultralytics, so the suite
runs without the baseline environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_training_slowdown import (
    ABBA_ORDER,
    DEFAULT_DATA,
    DEFAULT_MODEL,
    _block_layout,
    _pair_rows,
    _slowdown_percent,
    compose_payload,
    parse_args,
    resolve_config,
)


def _synthetic_blocks(seeds=(0, 1, 2), on_scale: float = 1.05) -> list[dict]:
    blocks = []
    for seed in seeds:
        base = 100.0 + float(seed)
        for block, arm in enumerate(ABBA_ORDER, start=1):
            seconds = base if arm == "OFF" else base * on_scale
            blocks.append(
                {
                    "seed": seed,
                    "block": block,
                    "arm": arm,
                    "epochs": 6,
                    "seconds": seconds,
                    "train_batches": 24,
                    "on_records": 0 if arm == "OFF" else 72,
                }
            )
    return blocks


def _payload(blocks=None, **overrides) -> dict:
    payload = compose_payload(
        run_id="run-training-slowdown",
        started_at="2026-09-10T00:00:00+08:00",
        finished_at="2026-09-10T01:00:00+08:00",
        model_config="ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml",
        data="ultralytics/cfg/datasets/coco8.yaml",
        seeds=[0, 1, 2],
        epochs_per_block=6,
        warmup_epochs=2,
        imgsz=640,
        batch=1,
        device="cpu",
        workers=2,
        blocks=_synthetic_blocks() if blocks is None else blocks,
        environment={"platform": "test", "python": "3.11", "torch": "2.13.0+cpu", "ultralytics": "8.4.101"},
    )
    payload.update(overrides)
    return payload


def test_block_layout_follows_frozen_abba_order() -> None:
    layout = _block_layout(epochs_per_block=6, warmup_epochs=2)
    assert [row["arm"] for row in layout] == ["OFF", "ON", "ON", "OFF"]
    assert layout[0]["block"] == 1 and layout[3]["block"] == 4
    assert layout[0]["epoch_start"] == 2
    assert layout[3]["epoch_end"] == 2 + 4 * 6
    # itertools.pairwise is 3.10+; the CI env is 3.9.
    for previous, following in zip(layout, layout[1:]):
        assert previous["epoch_end"] == following["epoch_start"]
    with pytest.raises(ValueError):
        _block_layout(epochs_per_block=0, warmup_epochs=2)


def test_slowdown_percent_math_and_guards() -> None:
    assert _slowdown_percent(100.0, 110.0) == pytest.approx(10.0)
    assert _slowdown_percent(100.0, 95.0) == pytest.approx(-5.0)
    for off, on in ((0.0, 1.0), (-1.0, 2.0), (1.0, float("nan"))):
        with pytest.raises(ValueError):
            _slowdown_percent(off, on)


def test_pair_rows_use_adjacent_abba_blocks() -> None:
    pairs = _pair_rows(_synthetic_blocks())
    assert len(pairs) == 6  # 3 seeds x 2 adjacent pairs
    for seed in (0, 1, 2):
        first = next(p for p in pairs if p["seed"] == seed and p["pair"] == 1)
        assert (first["off_block"], first["on_block"]) == (1, 2)
        assert first["slowdown_percent"] == pytest.approx(5.0)
        second = next(p for p in pairs if p["seed"] == seed and p["pair"] == 2)
        assert (second["off_block"], second["on_block"]) == (4, 3)  # ON(block3) vs OFF(block4)
        base = 100.0 + float(seed)
        assert second["slowdown_percent"] == pytest.approx((base * 1.05 - base) / base * 100.0)
    broken = [dict(row) for row in _synthetic_blocks(seeds=(0,))]
    broken[1]["arm"] = "OFF"
    with pytest.raises(ValueError):
        _pair_rows(broken)


def test_compose_payload_schema_and_verdict() -> None:
    payload = _payload()
    assert payload["schema_version"] == "e3-training-slowdown/v1"
    assert len(payload["blocks"]) == 12
    assert len(payload["pairs"]) == 6
    stats = payload["statistics"]["slowdown_percent"]
    assert stats["n"] == 6
    assert stats["mean"] == pytest.approx(5.0)
    assert len(stats["ci95"]) == 2
    assert stats["ci95"][0] <= stats["mean"] <= stats["ci95"][1]
    verdict = payload["verdict"]
    assert verdict["pass"] is True and verdict["ci95_upper_percent"] < 10.0
    assert payload["protocol"]["pairing"]["order"].startswith("OFF, ON, ON, OFF")
    assert "10.0%" in payload["protocol"]["threshold"]
    assert payload["parameters"]["seeds"] == [0, 1, 2]
    assert payload["parameters"]["epochs_per_seed"] == 2 + 4 * 6
    fail_payload = _payload(blocks=_synthetic_blocks(on_scale=1.2))
    assert fail_payload["verdict"]["pass"] is False
    assert fail_payload["verdict"]["ci95_upper_percent"] > 10.0


def test_resolve_config_validates_paths(tmp_path: Path) -> None:
    root = tmp_path / "baseline"
    model_file = root / DEFAULT_MODEL
    data_file = root / DEFAULT_DATA
    model_file.parent.mkdir(parents=True)
    data_file.parent.mkdir(parents=True)
    model_file.write_text("model: test\n", encoding="utf-8")
    data_file.write_text("path: coco8\n", encoding="utf-8")

    args = argparse.Namespace(
        baseline_root=str(root),
        model=DEFAULT_MODEL,
        data=DEFAULT_DATA,
        seeds="0,1,2",
        epochs_per_block=6,
        warmup_epochs=2,
        imgsz=640,
        batch=1,
        device="cpu",
        workers=2,
        run_id="run-check",
        output=None,
        dry_run=True,
        overwrite=False,
    )
    cfg = resolve_config(args)
    assert cfg["model"] == (root / DEFAULT_MODEL).resolve()
    assert cfg["data"] == (root / DEFAULT_DATA).resolve()
    assert cfg["seeds"] == [0, 1, 2]
    assert cfg["run_id"] == "run-check"
    assert cfg["output"].name == "slowdown_result.json"
    assert [row["arm"] for row in cfg["layout"]] == list(ABBA_ORDER)

    bad_seeds = argparse.Namespace(**{**vars(args), "seeds": "0,1,1"})
    with pytest.raises(ValueError):
        resolve_config(bad_seeds)

    data_file.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_config(args)


def test_parse_args_dry_run_round_trip() -> None:
    args = parse_args(["--dry-run", "--run-id", "run-x", "--epochs-per-block", "3"])
    assert args.dry_run is True
    assert args.run_id == "run-x"
    assert args.epochs_per_block == 3
