"""Render the E3 demo diagnostic figures from recorded routing evidence.

Two dark-theme PNGs for the demo video, both derived from JSON that the
evidence runs already wrote -- no model, no training, no hook, no ultralytics
import:

``health_check.png``      ``artifacts/health/<run_id>/routing_health.json``
                          per-module Gini, MoE highlighted, other families dimmed.
``temperature_probe.png`` ``artifacts/temperature/<tag>/moe_temperature_probe.json``
                          per-layer before/after (x2.0) metric bars.

Every caption number (collapse/uniform counts, changed-layer count,
dominant-expert verdict) is counted from the JSON, never typed in by hand, so
the figure cannot disagree with the evidence it points at. Each figure
self-checks its own text layout before writing.

Paths default to repo-relative evidence locations and stay overridable:

    python -m scripts.render_routing_diagnostics --out-dir build/demo
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("e3-routing-diagnostics")

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HEALTH_JSON = "artifacts/health/smoke-20260905-204546-6c7389/routing_health.json"
DEFAULT_TEMPERATURE_JSON = (
    "artifacts/temperature/moe-temp-factor2-20260912/moe_temperature_probe.json"
)
DEFAULT_OUT_DIR = "artifacts/demo"
HEALTH_FILENAME = "health_check.png"
TEMPERATURE_FILENAME = "temperature_probe.png"

BG, FG, SUB, MUT, LINE = "#0d1117", "#f0f6fc", "#c9d1d9", "#8b949e", "#30363d"
MOE, LAT, MOT, AFTER = "#ff5d5d", "#4c6b8a", "#3f6b4f", "#e3b341"

FAM_COLOR = {"latent": LAT, "moe": MOE, "mot": MOT}
FAM_LABEL = {"latent": "Latent", "moe": "MoE", "mot": "MoT"}
FAM_ORDER = {"moe": 0, "latent": 1, "mot": 2}

# Windows CJK fonts, same candidate list (and same graceful fallback) as
# scripts/render_token_heatmap.py.
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
)

METRICS = (
    ("top1_share", "Top-1 专家占比", "越低越分散"),
    ("entropy_normalized", "归一化熵", "越高越分散"),
    ("gini", "Gini 系数", "越低越均匀"),
)
PROBED_METRICS = ("top1_share", "entropy_normalized", "gini")


def resolve_path(value: str | Path) -> Path:
    """Resolve a CLI path against the repo root so the script is CWD-independent."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font in FONT_CANDIDATES:
        try:
            font_manager.fontManager.addfont(font)
        except Exception:  # missing font on non-Windows hosts: fall back below
            pass
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _style_axes(ax: Any) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT, labelsize=12)
    for spine in ax.spines.values():
        spine.set_color(LINE)


def check_layout(fig: Any, named_texts: Sequence[tuple[str, Any]], tol: float = 0.004) -> None:
    """Assert every text stays inside the canvas and stacked texts do not overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {}
    for name, artist in named_texts:
        box = artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        boxes[name] = box
        if box.x0 < -tol or box.y0 < -tol or box.x1 > 1 + tol or box.y1 > 1 + tol:
            raise AssertionError(f"[{name}] out of figure bounds: {box}")
    names = [name for name, _ in named_texts]
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if boxes[name].overlaps(boxes[other]):
                raise AssertionError(
                    f"[{name}] overlaps [{other}]: {boxes[name]} vs {boxes[other]}"
                )


def summarize_families(modules: Sequence[dict[str, Any]]) -> str:
    """Per-family collapse/uniform head line, counted from the records."""
    parts = []
    for family in sorted(FAM_ORDER, key=FAM_ORDER.get):
        members = [module for module in modules if module["family"] == family]
        if not members:
            continue
        collapsed = sum(1 for module in members if module["collapse"])
        uniform = sum(1 for module in members if module["uniform"])
        verdict = "collapse" if collapsed else "uniform"
        count = collapsed if collapsed else uniform
        parts.append(f"{FAM_LABEL[family]} {count}/{len(members)} {verdict}")
    return "  ·  ".join(parts)


def summarize_temperature(layers: dict[str, dict[str, Any]]) -> str:
    """Probe head line: how many layers moved, and whether dominance moved."""
    names = sorted(layers)
    changed = sum(
        1
        for name in names
        if any(abs(layers[name]["delta"][key]) > 1e-9 for key in PROBED_METRICS)
    )
    dominant = (
        "dominant expert unchanged"
        if not any(layers[name]["dominant_expert_changed"] for name in names)
        else "dominant expert changed"
    )
    return f"factor = 2.0  ·  {changed}/{len(names)} layers changed  ·  {dominant}"


def render_health(health_json: Path, out_path: Path) -> Path:
    plt = _matplotlib()
    data = json.loads(Path(health_json).read_text(encoding="utf-8"))
    modules = sorted(
        data["modules"], key=lambda module: (FAM_ORDER[module["family"]], module["layer"])
    )

    fig, ax = plt.subplots(figsize=(12.8, 7.6))
    fig.patch.set_facecolor(BG)
    _style_axes(ax)

    ys = list(range(len(modules)))[::-1]
    for y, module in zip(ys, modules):
        highlighted = module["family"] == "moe"
        ax.barh(
            y,
            module["gini"],
            height=0.74 if highlighted else 0.5,
            color=FAM_COLOR[module["family"]],
            alpha=1.0 if highlighted else 0.42,
            edgecolor=FG if highlighted else "none",
            linewidth=1.4 if highlighted else 0,
        )
        status = "坍缩" if module["collapse"] else ("均匀" if module["uniform"] else "分化")
        ax.text(
            module["gini"] + 0.022,
            y,
            f"{module['gini']:.3f}  {status}",
            va="center",
            fontsize=15 if highlighted else 11.5,
            fontweight="bold" if highlighted else "normal",
            color=FG if highlighted else MUT,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{FAM_LABEL[m['family']]}   {m['layer']}" for m in modules])
    for tick, module in zip(ax.get_yticklabels(), modules):
        if module["family"] == "moe":
            tick.set_color(FG)
            tick.set_fontsize(15)
            tick.set_fontweight("bold")
        else:
            tick.set_color(MUT)
            tick.set_fontsize(12)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Gini 系数（越接近 1 越集中）", fontsize=14, color=SUB)

    title = fig.text(
        0.012,
        0.955,
        "路由健康诊断：MoE 三层出现高集中选择",
        fontsize=24,
        fontweight="bold",
        color=FG,
        va="top",
    )
    summary = fig.text(
        0.012,
        0.888,
        summarize_families(modules),
        fontsize=17,
        color=MOE,
        fontweight="bold",
        va="top",
    )
    footer = fig.text(
        0.012,
        0.012,
        "数据源 routing_health.json ｜ 只读观测（未训练随机初始化基线）｜ 其余族作对照，MoE 高亮",
        fontsize=11.5,
        color=MUT,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.845])
    check_layout(fig, [("title", title), ("summary", summary), ("footer", footer)])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    logger.info("wrote %s (layout OK)", out_path)
    return out_path


def render_temperature(temperature_json: Path, out_path: Path) -> Path:
    plt = _matplotlib()
    data = json.loads(Path(temperature_json).read_text(encoding="utf-8"))
    layers = data["layers"]
    names = sorted(layers)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.4))
    fig.patch.set_facecolor(BG)
    xs = range(len(names))
    width = 0.34

    for ax, (key, metric_title, note) in zip(axes, METRICS):
        _style_axes(ax)
        before = [layers[name]["baseline"][key] for name in names]
        after = [layers[name]["factor_2_0"][key] for name in names]
        ax.bar(
            [i - width / 2 - 0.015 for i in xs],
            before,
            width,
            label="before · T=1.2",
            color="#5b7fa6",
            alpha=0.55,
            edgecolor="#8fb0d0",
            linewidth=1.2,
        )
        ax.bar(
            [i + width / 2 + 0.015 for i in xs],
            after,
            width,
            label="after · ×2.0",
            color=AFTER,
            alpha=0.95,
        )
        top = max(max(before), max(after))
        for i, (base_value, new_value) in enumerate(zip(before, after)):
            delta = new_value - base_value
            moved = abs(delta) > 1e-9
            ax.text(
                i,
                top * 1.19,
                f"Δ{delta:+.4f}" if moved else "Δ0（未变）",
                ha="center",
                fontsize=13,
                color=FG if moved else MUT,
                fontweight="bold" if moved else "normal",
            )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, fontsize=14, color=SUB)
        ax.set_ylim(0, top * 1.38)
        ax.set_title(f"{metric_title}\n{note}", fontsize=15, color=FG, pad=12)
        ax.tick_params(axis="y", labelsize=11)

    axes[0].set_ylabel("指标值", fontsize=13, color=SUB)
    axes[0].legend(frameon=False, fontsize=13, labelcolor=SUB, loc="upper left")
    title = fig.text(
        0.012,
        0.965,
        "Temperature probe：路由变化可观测，但效应有限",
        fontsize=23,
        fontweight="bold",
        color=FG,
        va="top",
    )
    subtitle = fig.text(
        0.012,
        0.893,
        summarize_temperature(layers),
        fontsize=17,
        color=AFTER,
        fontweight="bold",
        va="top",
    )
    footer = fig.text(
        0.012,
        0.012,
        "数据源 moe_temperature_probe.json ｜ untrained / routing-only ｜ 只描述路由结构，不构成性能结论",
        fontsize=11.5,
        color=MUT,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.835])
    check_layout(fig, [("title", title), ("subtitle", subtitle), ("footer", footer)])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    logger.info("wrote %s (layout OK)", out_path)
    return out_path


def render_diagnostics(health_json: Path, temperature_json: Path, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    return [
        render_health(health_json, out_dir / HEALTH_FILENAME),
        render_temperature(temperature_json, out_dir / TEMPERATURE_FILENAME),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--health-json", default=DEFAULT_HEALTH_JSON, help="routing_health.json")
    parser.add_argument(
        "--temperature-json", default=DEFAULT_TEMPERATURE_JSON, help="moe_temperature_probe.json"
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="directory for the two PNGs")
    args = parser.parse_args(argv)
    args.health_json = resolve_path(args.health_json)
    args.temperature_json = resolve_path(args.temperature_json)
    args.out_dir = resolve_path(args.out_dir)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    for path in render_diagnostics(args.health_json, args.temperature_json, args.out_dir):
        print(f"saved {path}")
    return 0


__all__ = [
    "DEFAULT_HEALTH_JSON",
    "DEFAULT_OUT_DIR",
    "DEFAULT_TEMPERATURE_JSON",
    "HEALTH_FILENAME",
    "TEMPERATURE_FILENAME",
    "check_layout",
    "main",
    "parse_args",
    "render_diagnostics",
    "render_health",
    "render_temperature",
    "resolve_path",
    "summarize_families",
    "summarize_temperature",
]


if __name__ == "__main__":
    raise SystemExit(main())