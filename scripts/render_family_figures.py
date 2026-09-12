"""Render the missing P0 static family figures (MoE, Latent) from run evidence.

The P0 static-figure set is one annotated heatmap per routing family. The MoT
figure (``mot_expert_heatmap_top1_share.png``) is produced by the upstream
``scripts/diagnose_mot_routing.py`` and copied into each run directory by
``run_e3_smoke.py``; MoE and Latent had no equivalent. This module fills that gap
by reading the evidence a finished run already wrote -- it never touches a
model, never registers a hook, and never imports ultralytics.

Reused plotting recipe
    The same matplotlib + seaborn annotated heatmap as the MoT figure (pivot,
    ``vmin=0, vmax=1``, 300 dpi, tight bbox), so the three figures read as one
    set. Only the data source differs.

Read-only sources
    MoE    ``<run_dir>/moe_usage_stats.json``  ExpertUsageTracker over coco8 val.
           rows = ``model.<idx>.routing``, cols = expert id, cell = selection share.
    Latent ``<run_dir>/routing_records.jsonl``  family == "latent" records.
           rows = LatentMixture module, cols = expert id, cell = expert_usage.

The MoT figure is deliberately NOT rendered here: it belongs to the upstream
script and must not be regenerated or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.routing_record import RoutingRecord

logger = logging.getLogger("e3-family-figures")

FIGURE_VERSION = "1.0.0"
GENERATOR = "scripts/render_family_figures.py"
SUPPORTED_FAMILIES: tuple[str, ...] = ("moe", "latent")
FIGURE_FILENAMES = {
    "moe": "moe_expert_selection_heatmap.png",
    "latent": "latent_expert_routing_heatmap.png",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_sort_key(name: str) -> tuple[int, str]:
    """Sort ``model.<n>[.routing]`` module names by their numeric index."""
    tail = name.split(".")[1] if "." in name else name
    try:
        return (int(tail), name)
    except ValueError:
        return (1 << 30, name)


def _resolve_run_id(run_dir: Path) -> str:
    summary = Path(run_dir) / "summary.json"
    if summary.is_file():
        try:
            run_id = json.loads(summary.read_text(encoding="utf-8")).get("run_id")
            if run_id:
                return str(run_id)
        except (json.JSONDecodeError, OSError):
            pass
    return Path(run_dir).name


def load_moe_matrix(run_dir: Path) -> tuple[list[str], list[int], list[list[float]]]:
    """Layer x expert selection-share matrix from ``moe_usage_stats.json``."""
    stats = json.loads((Path(run_dir) / "moe_usage_stats.json").read_text(encoding="utf-8"))
    rows = sorted(stats, key=_layer_sort_key)
    experts = sorted({int(expert) for layer in stats.values() for expert in layer})
    matrix: list[list[float]] = []
    for layer in rows:
        hits = {int(expert): float(entry["hits"]) for expert, entry in stats[layer].items()}
        total = sum(hits.values())
        matrix.append([hits.get(expert, 0.0) / total if total else 0.0 for expert in experts])
    return rows, experts, matrix


def load_latent_matrix(run_dir: Path) -> tuple[list[str], list[int], list[list[float]], dict[str, Any]]:
    """Module x expert routing-weight matrix from latent records plus routing meta."""
    path = Path(run_dir) / "routing_records.jsonl"
    records = [
        RoutingRecord.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latent = sorted((record for record in records if record.family == "latent"), key=lambda r: _layer_sort_key(r.module.name))
    if not latent:
        raise ValueError(f"no family='latent' records in {path}")
    experts = list(range(max(record.routing.num_experts for record in latent)))
    matrix = [
        [float(record.routing.expert_usage[i]) if i < len(record.routing.expert_usage) else 0.0 for i in experts]
        for record in latent
    ]
    head = latent[0]
    meta = {
        "dispatch_policy": head.family_data.get("dispatch_policy"),
        "top_k": head.routing.top_k,
        "routing_axis": head.family_data.get("routing_axis"),
        "routing_entropy_normalized": head.routing.routing_entropy_normalized,
        "dominant_expert_share": head.routing.dominant_expert_share,
    }
    return [record.module.name for record in latent], experts, matrix, meta


def render_heatmap(
    rows: Sequence[str],
    cols: Sequence[int],
    matrix: Sequence[Sequence[float]],
    *,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Annotated 0..1 heatmap using the same recipe as the upstream MoT figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    frame = pd.DataFrame([list(row) for row in matrix], index=list(rows), columns=[str(col) for col in cols])
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(cols) + 3.0), max(2.6, 0.5 * len(rows) + 2.0)))
    sns.heatmap(frame, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0, linewidths=0.4, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=16)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#4e5969")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_family_figures(
    run_dir: Path,
    out_dir: Path,
    *,
    families: Sequence[str] = SUPPORTED_FAMILIES,
) -> dict[str, Any]:
    """Render the MoE and/or Latent static figures for one finished run."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    run_id = _resolve_run_id(run_dir)
    figures: list[dict[str, Any]] = []
    for family in families:
        if family == "moe":
            rows, cols, matrix = load_moe_matrix(run_dir)
            source = run_dir / "moe_usage_stats.json"
            title = f"MoE expert selection share - run {run_id}"
            subtitle = (
                f"source: {source.name} (ExpertUsageTracker, coco8 val) | "
                f"{len(rows)} routing layers x {len(cols)} expert ids"
            )
            ylabel = "MoE layer (module.routing)"
            meta: dict[str, Any] = {"metric": "selection_share"}
        elif family == "latent":
            rows, cols, matrix, latent_meta = load_latent_matrix(run_dir)
            source = run_dir / "routing_records.jsonl"
            title = f"Latent expert routing weights - run {run_id}"
            subtitle = (
                f"source: {source.name} | dispatch={latent_meta['dispatch_policy']} "
                f"top_k={latent_meta['top_k']} axis={latent_meta['routing_axis']} | "
                f"normalized entropy={latent_meta['routing_entropy_normalized']:.3f}"
            )
            ylabel = "LatentMixture module"
            meta = {"metric": "expert_usage", **latent_meta}
        else:
            raise ValueError(f"unsupported family {family!r}; MoT is upstream-owned and never re-rendered here")
        out_path = out_dir / FIGURE_FILENAMES[family]
        render_heatmap(
            rows,
            cols,
            matrix,
            title=title,
            subtitle=subtitle,
            xlabel="expert id",
            ylabel=ylabel,
            out_path=out_path,
        )
        figures.append(
            {
                "family": family,
                "path": out_path.as_posix(),
                "source": source.as_posix(),
                "source_sha256": _sha256(source),
                "source_size_bytes": source.stat().st_size,
                "layers": list(rows),
                "experts": list(cols),
                **meta,
            }
        )
        logger.info("rendered %s figure: %s", family, out_path)
    metadata = {
        "figure_version": FIGURE_VERSION,
        "generator": GENERATOR,
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
        "families": list(families),
        "figures": figures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the MoE and Latent P0 static figures for a run.")
    parser.add_argument("--run-dir", required=True, help="artifacts/smoke/<run_id> directory")
    parser.add_argument("--out-dir", default="artifacts/figures/p0", help="output directory for the PNGs")
    parser.add_argument(
        "--families",
        default=",".join(SUPPORTED_FAMILIES),
        help=f"comma-separated families (default: {','.join(SUPPORTED_FAMILIES)}; MoT is excluded)",
    )
    args = parser.parse_args(argv)
    families = tuple(part.strip() for part in args.families.split(",") if part.strip())
    metadata = render_family_figures(Path(args.run_dir), Path(args.out_dir), families=families)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "FIGURE_FILENAMES",
    "FIGURE_VERSION",
    "GENERATOR",
    "SUPPORTED_FAMILIES",
    "load_latent_matrix",
    "load_moe_matrix",
    "render_family_figures",
    "render_heatmap",
]


if __name__ == "__main__":
    raise SystemExit(main())
