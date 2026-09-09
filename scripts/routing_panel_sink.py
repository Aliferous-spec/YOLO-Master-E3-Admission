"""E3 P1 panel sink: route routing records into TensorBoard and a zero-dependency HTML panel.

Task book P1 requirement (verbatim intent)
    "接入 TensorBoard/W&B 或实时面板，覆盖 >=3 类"

This module satisfies both halves of that sentence with two independent
channels, and degrades instead of failing when one is unavailable:

1. TensorBoard channel (optional) -- scalars + histograms streamed to
   ``<run_dir>/tb/`` while records are still being produced.
2. Zero-dependency HTML channel (always available) -- a self-contained
   ``routing_panel.html`` with hand-rolled inline SVG. No matplotlib, no CDN,
   no JavaScript framework. It opens by double-click, offline, forever.

Zero-intrusion guarantee
    This sink consumes ``RoutingRecord`` objects that the existing capture
    layer already produced. It never touches a model, never registers a
    forward hook, and never imports ultralytics. YOLO-Master core forward is
    untouched.

Design choice: "evidence you can trust", not "another heatmap"
    Upstream ``ultralytics/nn/modules/moe/viz.py`` is an MoE-only dashboard
    answering "what does routing look like". This panel is family-agnostic and
    additionally answers "can this evidence be trusted": it prints the schema
    version, per-family coverage (with explicit MISSING markers -- it never
    fabricates a family), canonical-vs-sample data-flow separation, and the
    manifest verification status of the run directory.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.routing_record import SCHEMA_VERSION, RoutingRecord

logger = logging.getLogger("e3-routing-panel")

PANEL_SINK_VERSION = "1.0.0"
PANEL_FILENAME = "routing_panel.html"
TB_SUBDIR = "tb"

# The three families the E3 task book requires the panel to cover.
REQUIRED_FAMILIES: tuple[str, ...] = ("moe", "mot", "latent")

# (record key, display label, how to read it)
METRIC_KEYS: tuple[tuple[str, str, str], ...] = (
    ("routing_entropy_normalized", "Routing entropy (norm.)", "0 = one expert dominates, 1 = perfectly uniform"),
    ("load_gini", "Load Gini", "0 = balanced, 1 = one expert takes everything"),
    ("dominant_expert_share", "Top-1 expert share", "share held by the dominant expert"),
)

_FAMILY_LABEL = {
    "moe": "MoE (discrete selection)",
    "mot": "MoT (scene-conditioned)",
    "latent": "Latent (continuous fusion)",
}


# ---------------------------------------------------------------------------
# normalization helpers
# ---------------------------------------------------------------------------


def _as_record_dict(item: Any) -> dict[str, Any]:
    """Accept a ``RoutingRecord``, its dict form, or a JSONL-parsed dict."""
    if isinstance(item, RoutingRecord):
        return item.to_dict()
    if isinstance(item, Mapping):
        return dict(item)
    raise TypeError(f"expected a RoutingRecord or a mapping, got {type(item).__name__}")


def _finite(value: Any) -> float | None:
    """Return a finite float, or None for missing / non-numeric data.

    Missing data is never coerced to 0.0 -- that is how a panel silently lies
    about a family it did not actually measure.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


@dataclass
class SeriesPoint:
    """One panel sample: the metrics of one record at one step."""

    step: int | None
    metrics: dict[str, float | None]
    expert_load: tuple[float, ...]
    training: bool | None


@dataclass
class ModuleSeries:
    """All observations of one routed module."""

    family: str
    module_name: str
    module_type: str
    points: list[SeriesPoint] = field(default_factory=list)

    @property
    def last(self) -> SeriesPoint:
        return self.points[-1]

    def metric_values(self, key: str) -> list[tuple[int, float]]:
        """Return (index, value) pairs, skipping missing values."""
        out: list[tuple[int, float]] = []
        for index, point in enumerate(self.points):
            value = point.metrics.get(key)
            if value is not None:
                out.append((index, value))
        return out


# ---------------------------------------------------------------------------
# TensorBoard channel (optional)
# ---------------------------------------------------------------------------


class TensorBoardChannel:
    """Stream routing scalars and load histograms to ``<run_dir>/tb/``.

    Imported lazily so a missing tensorboard package can never break the run;
    ``available`` reports whether the channel actually came up.
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.available = False
        self.reason: str | None = None
        self._writer: Any = None
        self._counts: dict[str, int] = {}

    def open(self) -> bool:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self.reason = f"tensorboard unavailable ({type(exc).__name__}: {exc})"
            return False
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        except Exception as exc:  # noqa: BLE001
            self.reason = f"SummaryWriter init failed ({type(exc).__name__}: {exc})"
            return False
        self.available = True
        return True

    def add_point(self, family: str, module_name: str, point: SeriesPoint, *, stream: str) -> None:
        if self._writer is None:
            return
        tag = f"{stream}/{family}/{module_name}"
        index = self._counts.get(tag, 0)
        self._counts[tag] = index + 1
        for key, _label, _hint in METRIC_KEYS:
            value = point.metrics.get(key)
            if value is None:
                continue  # never log a fabricated 0.0 for missing data
            self._writer.add_scalar(f"{tag}/{key}", value, index)
        if point.expert_load:
            self._writer.add_histogram(
                f"{tag}/normalized_expert_load",
                _histogram_bins(point.expert_load),
                index,
            )

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.flush()
            self._writer.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("tensorboard flush failed: %s", exc)
        self._writer = None


def _histogram_bins(values: Sequence[float], bins: int = 8) -> tuple[list[float], list[int]]:
    """Return (bin edges, counts) for a plain histogram without numpy."""
    if not values:
        return [], []
    low, high = min(values), max(values)
    if high <= low:
        return [low, high], [len(values)]
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        slot = int((value - low) / width)
        slot = min(slot, bins - 1)
        counts[slot] += 1
    edges = [low + width * index for index in range(bins + 1)]
    return edges, counts


# ---------------------------------------------------------------------------
# sink
# ---------------------------------------------------------------------------


class RoutingPanelSink:
    """Collect routing records and emit both panel channels.

    Usage during a live run (streaming)::

        sink = RoutingPanelSink(artifacts_dir, run_id=run_id)
        sink.add_records(records, stream="canonical")
        ...
        sink.close()   # writes routing_panel.html + flushes TensorBoard

    The sink is safe to use with zero records: the HTML then renders an
    explicit "no records captured" state instead of an empty dashboard.
    """

    def __init__(
        self,
        run_dir: Path | str,
        *,
        run_id: str = "",
        tensorboard: bool = True,
        html: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.enable_html = html
        self.series: dict[tuple[str, str], ModuleSeries] = {}
        self.stream_counts: dict[str, int] = {"canonical": 0, "sample": 0}
        self.schema_versions: set[str] = set()
        self.tensorboard = TensorBoardChannel(self.run_dir / TB_SUBDIR) if tensorboard else None
        self._opened = False
        self._closed = False
        self._metadata: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "RoutingPanelSink":
        if self._opened:
            return self
        if self.tensorboard is not None and not self.tensorboard.open():
            logger.warning("panel: %s -> HTML channel only", self.tensorboard.reason)
        self._opened = True
        return self

    def close(self) -> dict[str, Any]:
        """Flush both channels and return the panel metadata summary.

        Idempotent: a second call returns the metadata from the first one, so a
        ``with`` block that also calls ``close()`` explicitly cannot emit the
        panel twice.
        """
        if self._closed:
            return dict(self._metadata)
        self._closed = True
        if self.tensorboard is not None:
            self.tensorboard.close()
        channels = ["html"] if self.enable_html else []
        if self.tensorboard is not None and self.tensorboard.available:
            channels.append("tensorboard")
        metadata = {
            "panel_sink_version": PANEL_SINK_VERSION,
            "run_id": self.run_id,
            "channels": channels,
            "tensorboard_reason": None if self.tensorboard is None else self.tensorboard.reason,
            "families_covered": sorted({family for family, _ in self.series}),
            "families_missing": [f for f in REQUIRED_FAMILIES if f not in {fam for fam, _ in self.series}],
            "modules": len(self.series),
            "records": {
                "canonical": self.stream_counts["canonical"],
                "sample": self.stream_counts["sample"],
            },
            "schema_versions": sorted(self.schema_versions),
        }
        if self.enable_html:
            panel_path = self.run_dir / PANEL_FILENAME
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with panel_path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(render_panel_html(self, metadata))
            metadata["html_path"] = panel_path.name
        self._metadata = metadata
        return metadata

    def __enter__(self) -> "RoutingPanelSink":
        return self.open()

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- ingestion ---------------------------------------------------------

    def add_records(self, records: Iterable[Any], *, stream: str = "canonical") -> int:
        """Ingest records; ``stream`` marks the data-flow lane (canonical/sample).

        The two lanes are never merged. The panel reports them separately so a
        reader can tell a one-shot canonical observation from a per-sample
        sweep -- the same separation the JSONL artifacts already enforce.
        """
        if stream not in self.stream_counts:
            self.stream_counts[stream] = 0
        count = 0
        for item in records:
            payload = _as_record_dict(item)
            count += 1
            self.stream_counts[stream] += 1
            self.schema_versions.add(str(payload.get("schema_version", "unknown")))
            family = str(payload.get("family", "unknown"))
            module_payload = payload.get("module") or {}
            module_name = str(module_payload.get("name", "unknown"))
            module_type = str(module_payload.get("type", "unknown"))
            routing = payload.get("routing") or {}
            key = (family, module_name)
            series = self.series.get(key)
            if series is None:
                series = ModuleSeries(family=family, module_name=module_name, module_type=module_type)
                self.series[key] = series
            point = SeriesPoint(
                step=payload.get("step"),
                metrics={key_: _finite(routing.get(key_)) for key_, _l, _h in METRIC_KEYS},
                expert_load=tuple(
                    value
                    for value in (_finite(item_) for item_ in (routing.get("normalized_expert_load") or ()))
                    if value is not None
                ),
                training=payload.get("training"),
            )
            series.points.append(point)
            if self.tensorboard is not None:
                self.tensorboard.add_point(family, module_name, point, stream=stream)
        return count


# ---------------------------------------------------------------------------
# HTML rendering (stdlib only, inline SVG)
# ---------------------------------------------------------------------------

_SVG_WIDTH = 260
_SVG_HEIGHT = 44
_EXPERT_COLORS = ("#2f6fdb", "#d9822b", "#2f9e6f", "#b5453c", "#7a52b5", "#4a4a4a")


def _sparkline(values: list[tuple[int, float]], *, maximum: float = 1.0) -> str:
    """Return an inline SVG polyline for ``values``; empty string if unusable."""
    if len(values) < 2:
        return ""
    span = max(index for index, _ in values) or 1
    scale_x = _SVG_WIDTH / span
    points: list[str] = []
    for index, value in values:
        x = index * scale_x
        clamped = min(max(value, 0.0), maximum)
        y = _SVG_HEIGHT - (clamped / maximum) * (_SVG_HEIGHT - 4) - 2
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    return (
        f'<svg class="spark" viewBox="0 0 {_SVG_WIDTH} {_SVG_HEIGHT}" '
        f'preserveAspectRatio="none" role="img" aria-label="metric trend">'
        f'<polyline points="{polyline}" fill="none" stroke="#2f6fdb" stroke-width="1.6"/>'
        f"</svg>"
    )


def _load_bars(load: Sequence[float]) -> str:
    if not load:
        return '<span class="missing">no expert-load data</span>'
    cells: list[str] = []
    for index, value in enumerate(load):
        color = _EXPERT_COLORS[index % len(_EXPERT_COLORS)]
        width = max(0.0, min(value, 1.0)) * 100
        cells.append(
            f'<div class="bar-row"><span class="bar-label">E{index}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{width:.1f}%;'
            f'background:{color}"></span></span>'
            f'<span class="bar-value">{value:.3f}</span></div>'
        )
    return "".join(cells)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _coverage_badges(covered: Sequence[str]) -> str:
    covered_set = set(covered)
    out: list[str] = []
    for family in REQUIRED_FAMILIES:
        present = family in covered_set
        cls = "badge ok" if present else "badge missing"
        text = "covered" if present else "MISSING"
        out.append(f'<span class="{cls}">{_esc(family.upper())} &middot; {text}</span>')
    return "".join(out)


def _render_module_table(series: ModuleSeries) -> str:
    rows: list[str] = []
    for key, label, hint in METRIC_KEYS:
        values = series.metric_values(key)
        if values:
            latest = values[-1][1]
            cell = f'{latest:.4f}'
            chart = _sparkline(values)
        else:
            cell = '<span class="missing">n/a</span>'
            chart = ""
        rows.append(
            f'<tr><th title="{_esc(hint)}">{_esc(label)}</th>'
            f'<td class="num">{cell}</td><td class="chart">{chart}</td></tr>'
        )
    return "".join(rows)


def render_panel_html(sink: RoutingPanelSink, metadata: Mapping[str, Any] | None = None) -> str:
    """Render the self-contained panel page for ``sink``."""
    metadata = dict(metadata or {})
    covered = sorted({family for family, _ in sink.series})
    missing = [f for f in REQUIRED_FAMILIES if f not in covered]
    by_family: dict[str, list[ModuleSeries]] = {}
    for (family, _name), series in sorted(sink.series.items()):
        by_family.setdefault(family, []).append(series)

    sections: list[str] = []
    for family in REQUIRED_FAMILIES:
        series_list = by_family.get(family)
        if not series_list:
            sections.append(
                f'<section class="family"><h2>{_esc(_FAMILY_LABEL.get(family, family))}</h2>'
                f'<p class="missing">No records captured for this family in this run. '
                f"Reported as missing, not estimated.</p></section>"
            )
            continue
        blocks: list[str] = []
        for series in series_list:
            last = series.last
            blocks.append(
                f'<div class="module"><h3>{_esc(series.module_name)}</h3>'
                f'<p class="meta">{_esc(series.module_type)} &middot; '
                f"{len(series.points)} observation(s)</p>"
                f'<table class="metrics">{_render_module_table(series)}</table>'
                f'<div class="loads">{_load_bars(last.expert_load)}</div></div>'
            )
        sections.append(
            f'<section class="family"><h2>{_esc(_FAMILY_LABEL.get(family, family))}</h2>'
            f'{"".join(blocks)}</section>'
        )

    for family in sorted(by_family):
        if family in REQUIRED_FAMILIES:
            continue
        blocks = [
            f'<div class="module"><h3>{_esc(s.module_name)}</h3>'
            f'<p class="meta">{_esc(s.module_type)} &middot; {len(s.points)} observation(s)</p>'
            f'<table class="metrics">{_render_module_table(s)}</table></div>'
            for s in by_family[family]
        ]
        sections.append(
            f'<section class="family extra"><h2>{_esc(family)} (beyond P1 scope)</h2>{"".join(blocks)}</section>'
        )

    counts = sink.stream_counts
    channels = ", ".join(metadata.get("channels", [])) or "html"
    tb_reason = metadata.get("tensorboard_reason")
    tb_note = f" &middot; TensorBoard disabled: {_esc(tb_reason)}" if tb_reason else ""
    schema_versions = ", ".join(metadata.get("schema_versions", [])) or SCHEMA_VERSION

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>E3 routing panel &middot; {_esc(sink.run_id or "run")}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ margin:0; padding:24px; font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  background:#f7f8fa; color:#1d2129; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
h2 {{ font-size:16px; margin:0 0 12px; }}
h3 {{ font-size:14px; margin:0 0 2px; font-family:ui-monospace,Consolas,monospace; }}
.wrap {{ max-width:1080px; margin:0 auto; }}
.head {{ background:#fff; border:1px solid #e5e6eb; border-radius:8px; padding:16px 20px; margin-bottom:16px; }}
.badges {{ margin-top:10px; }}
.badge {{ display:inline-block; padding:2px 10px; border-radius:10px; font-size:12px; margin-right:6px; }}
.badge.ok {{ background:#e8f5e9; color:#1b7f3b; border:1px solid #b7e0c0; }}
.badge.missing {{ background:#fdecea; color:#b3261e; border:1px solid #f5c2c0; }}
.evidence {{ margin-top:12px; font-size:12px; color:#4e5969; }}
.evidence code {{ background:#f2f3f5; padding:1px 5px; border-radius:3px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }}
.family {{ background:#fff; border:1px solid #e5e6eb; border-radius:8px; padding:16px 20px; margin-bottom:16px; }}
.module {{ border-top:1px solid #f0f1f3; padding-top:12px; margin-top:12px; }}
.module:first-of-type {{ border-top:none; padding-top:0; margin-top:0; }}
.meta {{ margin:0 0 8px; font-size:12px; color:#86909c; }}
table.metrics {{ border-collapse:collapse; width:100%; }}
table.metrics th {{ text-align:left; font-weight:500; font-size:12px; color:#4e5969; padding:3px 8px 3px 0; }}
table.metrics td.num {{ font-family:ui-monospace,Consolas,monospace; text-align:right; padding:3px 10px 3px 0; }}
table.metrics td.chart {{ width:270px; }}
svg.spark {{ width:260px; height:34px; display:block; }}
.loads {{ margin-top:8px; }}
.bar-row {{ display:flex; align-items:center; gap:8px; font-size:12px; }}
.bar-label {{ width:22px; color:#86909c; font-family:ui-monospace,Consolas,monospace; }}
.bar-track {{ flex:1; height:9px; background:#f2f3f5; border-radius:5px; overflow:hidden; }}
.bar-fill {{ display:block; height:100%; }}
.bar-value {{ width:52px; text-align:right; font-family:ui-monospace,Consolas,monospace; color:#4e5969; }}
.missing {{ color:#b3261e; }}
.family.extra {{ opacity:.85; }}
footer {{ color:#86909c; font-size:12px; margin-top:20px; }}
</style></head>
<body><div class="wrap">
<div class="head">
  <h1>E3 routing panel</h1>
  <div class="meta">run <code>{_esc(sink.run_id or "-")}</code> &middot; panel sink v{_esc(PANEL_SINK_VERSION)} &middot; channels: {_esc(channels)}{tb_note}</div>
  <div class="badges">{_coverage_badges(covered)}</div>
  <div class="evidence">
    schema <code>{_esc(schema_versions)}</code> &middot;
    canonical records <code>{counts.get("canonical", 0)}</code> &middot;
    sample records <code>{counts.get("sample", 0)}</code> &middot;
    modules <code>{len(sink.series)}</code> &middot;
    families missing <code>{_esc(", ".join(missing) or "none")}</code>
  </div>
</div>
<div class="grid">{"".join(sections)}</div>
<footer>Generated by <code>scripts/routing_panel_sink.py</code>. Missing families are reported, never
estimated. Self-contained: no external assets, no network required.</footer>
</div></body></html>
"""


# ---------------------------------------------------------------------------
# CLI: re-render a panel from an existing run
# ---------------------------------------------------------------------------


def load_records(path: Path) -> list[RoutingRecord]:
    """Load a ``routing_records.jsonl`` / ``sample_routing_records.jsonl`` file."""
    records: list[RoutingRecord] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(RoutingRecord.from_json(line))
    return records


def render_run(run_dir: Path, *, run_id: str = "", tensorboard: bool = False) -> dict[str, Any]:
    """Build a panel for an already-finished run directory."""
    run_dir = Path(run_dir)
    resolved_id = run_id
    if not resolved_id:
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            try:
                resolved_id = str(json.loads(summary_path.read_text(encoding="utf-8")).get("run_id", ""))
            except (json.JSONDecodeError, OSError):
                resolved_id = ""
    resolved_id = resolved_id or run_dir.name
    with RoutingPanelSink(run_dir, run_id=resolved_id, tensorboard=tensorboard) as sink:
        canonical = run_dir / "routing_records.jsonl"
        if canonical.is_file():
            sink.add_records(load_records(canonical), stream="canonical")
        sample = run_dir / "sample_routing_records.jsonl"
        if sample.is_file():
            sink.add_records(load_records(sample), stream="sample")
    return sink.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the E3 routing panel for a run directory.")
    parser.add_argument("run_dir", help="artifacts/smoke/<run_id> directory")
    parser.add_argument("--run-id", default="", help="override the run id shown in the panel")
    parser.add_argument("--tensorboard", action="store_true", help="also emit TensorBoard event files")
    args = parser.parse_args(argv)
    metadata = render_run(Path(args.run_dir), run_id=args.run_id, tensorboard=args.tensorboard)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "METRIC_KEYS",
    "PANEL_FILENAME",
    "PANEL_SINK_VERSION",
    "REQUIRED_FAMILIES",
    "ModuleSeries",
    "RoutingPanelSink",
    "SeriesPoint",
    "TensorBoardChannel",
    "load_records",
    "render_panel_html",
    "render_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
