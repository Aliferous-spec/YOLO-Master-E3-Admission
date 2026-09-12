"""Assemble the E3 demo video from already-rendered PNG evidence.

1920x1080 / 24 fps / 100 s / H.264, no audio track, everything burned into the
picture. Frames are drawn with PIL and piped straight into ffmpeg as raw BGR --
no intermediate video, no cv2.

Inputs are read-only and repo-relative by default:

    assets (figures/diagnostic PNGs)  artifacts/demo
      token_heatmap_000000000042.jpg.png   (from artifacts/figures/p2)
      token_heatmap_000000000049.jpg.png   (from artifacts/figures/p2)
      health_check.png                     (scripts/render_routing_diagnostics.py)
      temperature_probe.png                (scripts/render_routing_diagnostics.py)

    python -m scripts.build_demo_video --assets-dir artifacts/demo --out build/demo.mp4

ffmpeg comes from --ffmpeg, else $E3_FFMPEG, else $PATH. --duration-scale
shrinks every segment (useful for a fast preview / encode smoke check) without
touching the shipped 100 s timing, which stays the default.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("e3-demo-video")

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ASSETS_DIR = "artifacts/demo"
DEFAULT_FIGURES_DIR = "artifacts/figures/p2"
DEFAULT_OUT = "artifacts/demo/e3-routing-demo-h264.mp4"

WIDTH, HEIGHT, FPS = 1920, 1080, 24
BG = (13, 17, 23)
FG = (240, 246, 252)
SUB = (201, 209, 217)
MUT = (139, 148, 158)
ACCENT = (88, 166, 255)
GOLD = (227, 179, 65)

# Same Windows CJK candidates as scripts/render_token_heatmap.py; PIL needs a
# real file path, so --font / --font-title can override them.
FONT_BODY_CANDIDATES = (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf")
FONT_TITLE_CANDIDATES = (
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
)

# (segment builder name, frame count) at --duration-scale 1.0 -- 2400 frames = 100 s.
SEGMENT_FRAMES = {
    "title": 192,
    "scan": 192,
    "heatmap_a": 600,
    "heatmap_b": 480,
    "health": 360,
    "temperature": 360,
    "limitations": 216,
}
TOTAL_FRAMES = sum(SEGMENT_FRAMES.values())

HEATMAP_A = "token_heatmap_000000000042.jpg.png"
HEATMAP_B = "token_heatmap_000000000049.jpg.png"
HEALTH_PNG = "health_check.png"
TEMPERATURE_PNG = "temperature_probe.png"


def resolve_path(value: str | Path) -> Path:
    """Resolve a CLI path against the repo root so the script is CWD-independent."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_ffmpeg(value: str | None) -> str:
    """Locate the ffmpeg executable: --ffmpeg, else $E3_FFMPEG, else $PATH."""
    candidate = value or os.environ.get("E3_FFMPEG") or "ffmpeg"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"ffmpeg not found: {candidate!r} (pass --ffmpeg PATH or set E3_FFMPEG)"
        )
    return resolved


def resolve_font(candidates: Sequence[str], override: str | None) -> str:
    for candidate in ([override] if override else []) + list(candidates):
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(f"no usable font found; tried {list(candidates)} (use --font)")


def _require(path: Path) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(f"missing demo input: {path}")
    return Path(path)


def load_assets(assets_dir: Path, figures_dir: Path) -> dict[str, Any]:
    """Load the four PNGs the video is built from, failing loudly on a gap."""
    from PIL import Image

    assets_dir, figures_dir = Path(assets_dir), Path(figures_dir)
    return {
        "heatmap_a": Image.open(_require(figures_dir / HEATMAP_A)).convert("RGB"),
        "heatmap_b": Image.open(_require(figures_dir / HEATMAP_B)).convert("RGB"),
        "health": Image.open(_require(assets_dir / HEALTH_PNG)).convert("RGB"),
        "temperature": Image.open(_require(assets_dir / TEMPERATURE_PNG)).convert("RGB"),
    }


def build_segments(
    images: dict[str, Any], fonts: dict[str, Any], duration_scale: float = 1.0
) -> list[tuple[Callable[[float], Any], int]]:
    """Frame builders in playback order, with their frame counts."""
    from PIL import Image, ImageDraw

    def frame() -> tuple[Any, Any]:
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        return image, ImageDraw.Draw(image)

    def centered(draw: Any, text: str, font: Any, y: int, fill: Any = FG) -> None:
        width = draw.textbbox((0, 0), text, font=font)[2]
        draw.text(((WIDTH - width) // 2, y), text, font=font, fill=fill)

    def fit(image: Any, max_w: int = 1800, max_h: int = 790) -> Any:
        iw, ih = image.size
        scale = min(max_w / iw, max_h / ih)
        return image.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)

    def zoom(image: Any, scale: float) -> Any:
        iw, ih = image.size
        big = image.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
        x, y = (big.size[0] - iw) // 2, (big.size[1] - ih) // 2
        return big.crop((x, y, x + iw, y + ih))

    def blend(image: Any, alpha: float) -> Any:
        if alpha >= 1.0:
            return image
        return Image.blend(Image.new("RGB", image.size, BG), image, alpha)

    def fade(index: int, total: int, ramp: int = 14) -> float:
        if index < ramp:
            return index / ramp
        if index > total - ramp:
            return max(0.0, (total - index) / ramp)
        return 1.0

    def seg_title(_t: float) -> Any:
        image, draw = frame()
        centered(draw, "E3 · 腾讯犀牛鸟实战课题", fonts["note"], 176, GOLD)
        centered(draw, "路由可视化 Demo", fonts["big"], 300)
        centered(draw, "YOLO-Master · MoT / MoE / Latent", fonts["h2"], 456, SUB)
        draw.rectangle([760, 566, 1160, 569], fill=ACCENT)
        centered(draw, "P2：空间路由可视化", fonts["h1"], 624, GOLD)
        centered(draw, "零侵入 · 只读观测 · 不改 forward", fonts["cap"], 716, MUT)
        centered(draw, "作者：刘欣然 ｜ 广州应用科技学院", fonts["small"], 936, MUT)
        return image

    def seg_image(image: Any, title: str, caption: str, t: float, mode: str = "static") -> Any:
        out, draw = frame()
        centered(draw, title, fonts["h2"], 46)
        centered(draw, caption, fonts["cap_bold"], HEIGHT - 92, SUB)
        source = zoom(image, 1.0 + 0.05 * t) if mode == "kenburns" else image
        fitted = fit(source)
        iw, ih = fitted.size
        x, y = (WIDTH - iw) // 2, 560 - ih // 2
        out.paste(fitted, (x, y))
        if mode == "scan":
            column_width = iw / 4.0
            column = 3 if t >= 1.0 else int(t * 3.999)
            hx = int(x + column * column_width)
            draw.rectangle([hx, y, int(hx + column_width), y + ih], outline=GOLD, width=6)
        return out

    def seg_limitations(_t: float) -> Any:
        image, draw = frame()
        centered(draw, "局限与边界", fonts["big"], 108)
        lines = (
            ("untrained / routing-only：只描述路由结构，不构成性能结论", GOLD),
            ("未训练随机初始化基线，非训练性能结论", SUB),
            ("全程不修改 forward，纯只读观测", SUB),
            ("本轮只做 MoE 空间路由；MoT / Latent 无同类 token 级路由面", SUB),
            ("P2 属超出初始 P1 范围的新增成果", MUT),
        )
        y = 300
        for text, color in lines:
            centered(draw, text, fonts["cap"], y, color)
            y += 74
        draw.rectangle([760, 700, 1160, 702], fill=ACCENT)
        centered(
            draw,
            "github.com/Aliferous-spec/YOLO-Master-E3-Admission · PR #1",
            fonts["small"],
            748,
            MUT,
        )
        centered(
            draw,
            "run-id smoke-20260905-204546-6c7389 ｜ moe-temp-factor2-20260912",
            fonts["tiny"],
            794,
            MUT,
        )
        centered(draw, "谢谢 ｜ 欢迎讨论", fonts["h2"], 872)
        return image

    named: list[tuple[str, Callable[[float], Any]]] = [
        ("title", seg_title),
        (
            "scan",
            lambda t: seg_image(
                images["heatmap_b"],
                "输入图像 → MoE 路由热图",
                "颜色 = dominant expert ID（不表示概率大小）",
                t,
                "scan",
            ),
        ),
        (
            "heatmap_a",
            lambda t: seg_image(
                images["heatmap_a"],
                "MoE 空间路由：model.5 / model.8 / model.11",
                "专家数 4 → 8 → 16 ｜ 颜色 = dominant expert ID",
                t,
                "kenburns",
            ),
        ),
        (
            "heatmap_b",
            lambda t: seg_image(
                images["heatmap_b"],
                "同一结论，换一张 COCO8 图像",
                "空间上成片接管 ｜ 非路由概率大小",
                t,
                "kenburns",
            ),
        ),
        (
            "health",
            lambda t: seg_image(
                images["health"],
                "路由健康诊断：MoE 三层出现高集中选择",
                "MoE 3/3 collapse · Latent 3/3 uniform · MoT 3/9 collapse",
                t,
            ),
        ),
        (
            "temperature",
            lambda t: seg_image(
                images["temperature"],
                "Temperature probe：路由变化可观测，但效应有限",
                "factor = 2.0 · 2/3 layers changed · dominant expert unchanged",
                t,
            ),
        ),
        ("limitations", seg_limitations),
    ]

    segments = []
    for name, builder in named:
        count = max(1, round(SEGMENT_FRAMES[name] * duration_scale))
        wrapped = (lambda builder, count: lambda i: blend(builder(i / count), fade(i, count)))(
            builder, count
        )
        segments.append((wrapped, count))
    return segments


def load_fonts(body_path: str, title_path: str) -> dict[str, Any]:
    from PIL import ImageFont

    def font(path: str, size: int) -> Any:
        return ImageFont.truetype(path, size)

    return {
        "big": font(title_path, 88),
        "h1": font(title_path, 62),
        "h2": font(title_path, 52),
        "cap_bold": font(title_path, 38),
        "cap": font(body_path, 36),
        "note": font(body_path, 30),
        "small": font(body_path, 28),
        "tiny": font(body_path, 25),
    }


def encode_video(
    segments: Sequence[tuple[Callable[[float], Any], int]],
    out_path: Path,
    ffmpeg: str,
    fps: int = FPS,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> int:
    """Pipe raw BGR frames into ffmpeg and wait for it; returns ffmpeg's exit code."""
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-g", "48", "-movflags", "+faststart", "-an", str(out_path),
    ]
    frames = sum(count for _, count in segments)
    logger.info("encoding %d frames (%.2f s) -> %s", frames, frames / fps, out_path)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    written = 0
    for builder, count in segments:
        for index in range(count):
            frame = builder(index)
            pixels = np.asarray(frame.convert("RGB"), dtype=np.uint8)[..., ::-1]
            process.stdin.write(pixels.tobytes())
            written += 1
        logger.info("segment done (%d frames)", count)
    process.stdin.close()
    code = process.wait()
    logger.info("ffmpeg exit %d, %d frames written", code, written)
    return code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR, help="diagnostic PNGs")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR, help="P2 token heatmaps")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output mp4 path")
    parser.add_argument(
        "--ffmpeg", default=None, help="ffmpeg executable (default: $E3_FFMPEG, $PATH)"
    )
    parser.add_argument("--font", default=None, help="body font file")
    parser.add_argument("--font-title", default=None, help="title font file")
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="scale every segment (default 1.0 = 2400 frames / 100 s)",
    )
    args = parser.parse_args(argv)
    args.assets_dir = resolve_path(args.assets_dir)
    args.figures_dir = resolve_path(args.figures_dir)
    args.out = resolve_path(args.out)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    body = resolve_font(FONT_BODY_CANDIDATES, args.font)
    title = resolve_font(FONT_TITLE_CANDIDATES, args.font_title)
    images = load_assets(args.assets_dir, args.figures_dir)
    segments = build_segments(images, load_fonts(body, title), args.duration_scale)
    return encode_video(segments, args.out, ffmpeg)


__all__ = [
    "DEFAULT_ASSETS_DIR",
    "DEFAULT_FIGURES_DIR",
    "DEFAULT_OUT",
    "HEIGHT",
    "SEGMENT_FRAMES",
    "TOTAL_FRAMES",
    "WIDTH",
    "build_segments",
    "encode_video",
    "load_assets",
    "load_fonts",
    "main",
    "parse_args",
    "resolve_ffmpeg",
    "resolve_font",
    "resolve_path",
]


if __name__ == "__main__":
    sys.exit(main())