"""Tests for the demo video builder.

The real encode needs ffmpeg; when it is not installed (e.g. the default CI
image) only the cheap plan/path tests run, and the encode test skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageFont

from scripts.build_demo_video import (
    HEIGHT,
    REPO_ROOT,
    SEGMENT_FRAMES,
    TOTAL_FRAMES,
    WIDTH,
    build_segments,
    main,
    parse_args,
    resolve_ffmpeg,
    resolve_font,
    resolve_path,
)

FTYP = b"ftyp"


def _images() -> dict:
    return {
        name: Image.new("RGB", (320, 200), (13, 17, 23))
        for name in ("heatmap_a", "heatmap_b", "health", "temperature")
    }


def _fonts() -> dict:
    font = ImageFont.load_default()
    return {name: font for name in ("big", "h1", "h2", "cap_bold", "cap", "note", "small", "tiny")}


def _write_demo_assets(root: Path) -> tuple[Path, Path]:
    assets_dir = root / "assets"
    figures_dir = root / "figures"
    assets_dir.mkdir()
    figures_dir.mkdir()
    for name in ("health_check.png", "temperature_probe.png"):
        Image.new("RGB", (640, 400), (13, 17, 23)).save(assets_dir / name)
    for name in ("token_heatmap_000000000042.jpg.png", "token_heatmap_000000000049.jpg.png"):
        Image.new("RGB", (640, 400), (13, 17, 23)).save(figures_dir / name)
    return assets_dir, figures_dir


def test_parse_args_resolves_repo_relative_paths() -> None:
    args = parse_args([])

    assert args.assets_dir == REPO_ROOT / "artifacts/demo"
    assert args.figures_dir == REPO_ROOT / "artifacts/figures/p2"
    assert args.out == REPO_ROOT / "artifacts/demo/e3-routing-demo-h264.mp4"
    assert args.duration_scale == 1.0


def test_parse_args_honours_explicit_paths(tmp_path: Path) -> None:
    assert resolve_path(tmp_path / "a.mp4") == tmp_path / "a.mp4"
    assert resolve_path("build/a.mp4") == REPO_ROOT / "build" / "a.mp4"

    args = parse_args(
        [
            "--assets-dir", str(tmp_path),
            "--figures-dir", "figs",
            "--out", "build/demo.mp4",
            "--ffmpeg", "custom-ffmpeg",
            "--duration-scale", "0.5",
        ]
    )

    assert args.assets_dir == tmp_path
    assert args.figures_dir == REPO_ROOT / "figs"
    assert args.out == REPO_ROOT / "build" / "demo.mp4"
    assert args.ffmpeg == "custom-ffmpeg"
    assert args.duration_scale == 0.5


def test_segment_plan_is_the_shipped_100s_timeline() -> None:
    assert TOTAL_FRAMES == 2400
    assert sum(SEGMENT_FRAMES.values()) == TOTAL_FRAMES

    full = build_segments(_images(), _fonts(), 1.0)
    scaled = build_segments(_images(), _fonts(), 0.02)

    assert sum(count for _, count in full) == 2400
    assert sum(count for _, count in scaled) == 48
    assert len(full) == len(scaled) == len(SEGMENT_FRAMES)

    frame = full[0][0](0.5)
    assert frame.size == (WIDTH, HEIGHT)


def test_resolve_font_prefers_the_override_and_fails_loudly(tmp_path: Path) -> None:
    real = tmp_path / "font.ttf"
    real.write_bytes(Path(__file__).read_bytes())

    assert resolve_font(("nope.ttf",), str(real)) == str(real)
    with pytest.raises(FileNotFoundError):
        resolve_font((str(tmp_path / "missing.ttf"),), None)


def test_main_encodes_a_preview_mp4(tmp_path: Path) -> None:
    try:
        ffmpeg = resolve_ffmpeg(None)
    except FileNotFoundError:
        pytest.skip("ffmpeg not available")

    assets_dir, figures_dir = _write_demo_assets(tmp_path)
    out = tmp_path / "preview.mp4"

    assert main(
        [
            "--assets-dir", str(assets_dir),
            "--figures-dir", str(figures_dir),
            "--out", str(out),
            "--ffmpeg", ffmpeg,
            "--duration-scale", "0.01",
        ]
    ) == 0
    payload = out.read_bytes()
    assert len(payload) > 0
    assert payload[4:8] == FTYP