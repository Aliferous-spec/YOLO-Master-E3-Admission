"""E3 P2 slice 1: MoE token-routing heatmap overlaid on the original COCO8 image.

Read-only observation of the deployed YOLO-Master baseline:

- the upstream forward is never touched: the MoE routing module and its spatial
  branch are only observed through ``register_forward_hook``;
- no upstream code, schema, config, protocol or existing artifact is modified;
- the only new outputs are PNGs plus one provenance JSON under
  ``artifacts/figures/p2/``.

Measured contract on the deployed baseline (``yolo-master-n``, v0.9):
``DetailAwareLowRankHybridAdaptiveGateMoE`` (``model.5`` / ``model.8`` /
``model.11``) routes through ``DualStreamGateRouter``, which returns
``(routing_weights[B, top_k, 1, 1], routing_indices[B, top_k, 1, 1], routing_stats)``.
``routing_stats["router_probs"]`` is published only in train mode and is
**per image** ``[B, E]``: the router mean-pools its spatial branch
(``local_conv`` output ``[B, E, h, w]``) before the softmax.  The published
routing decision is therefore per-image, and the only per-token routing signal
this model exposes is that spatial branch.

Both layouts are handled explicitly:

- rank-4 ``router_probs`` -> used as-is as per-token ``[B, E, H, W]`` probs;
- rank-2 per-image ``router_probs`` plus captured spatial-branch logits ->
  per-token softmax over experts, tagged ``router_spatial_branch_softmax``.

MoE only: MoT / Latent expose no comparable per-token routing surface and are
deliberately not rendered here.

Example:
    python scripts/render_token_heatmap.py --config configs/e3_smoke.yaml \
        --baseline-root C:/path/to/YOLO-Master
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

logger = logging.getLogger("e3-token-heatmap")

FIGURE_VERSION = "1.1.0"
GENERATOR = "scripts/render_token_heatmap.py"
DEFAULT_ALPHA = 0.45
DEFAULT_CMAP = "turbo"
NON_TENSOR_KEYS = ("token_probs", "token_ids")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_router_probs(output: Any) -> torch.Tensor:
    """Return ``router_probs`` from a routing-module forward-hook ``output``."""
    stats: Mapping[str, Any] | None = None
    candidates = output if isinstance(output, tuple) else (output,)
    for item in candidates:
        if isinstance(item, Mapping):
            stats = item
            break
    if stats is None:
        raise ValueError(
            "routing module output carries no routing-stats mapping; expected "
            "(routing_weights, routing_indices, routing_stats)"
        )
    probs = stats.get("router_probs")
    if probs is None:
        raise ValueError(
            "routing stats have no 'router_probs' (the upstream router publishes it only in train mode)"
        )
    if not isinstance(probs, torch.Tensor):
        raise TypeError(f"router_probs must be a Tensor, got {type(probs).__name__}")
    return probs.detach()


def resolve_token_probs(
    router_probs: torch.Tensor | None,
    spatial_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    """Reduce a router output to per-token ``[B, E, h, w]`` probabilities.

    Rank-4 ``router_probs`` are already per-token and are used as-is.  A rank-2
    per-image ``[B, E]`` tensor is only usable together with the router's
    captured spatial-branch logits, which are softmaxed over experts per token.
    """
    if router_probs is None:
        raise ValueError("no router_probs captured for this routing module")
    if router_probs.ndim == 4:
        return router_probs, "router_probs"
    if router_probs.ndim == 2:
        if spatial_logits is None:
            raise ValueError(
                f"router_probs is per-image {tuple(router_probs.shape)} (no token axis) and no router "
                "spatial branch was captured; this router exposes no per-token probabilities"
            )
        if spatial_logits.ndim != 4:
            raise ValueError(
                f"router spatial branch must be [B, E, h, w], got {tuple(spatial_logits.shape)}"
            )
        return torch.softmax(spatial_logits.float(), dim=1).detach(), "router_spatial_branch_softmax"
    raise ValueError(f"unsupported router_probs rank {router_probs.ndim}: {tuple(router_probs.shape)}")


def dominant_expert_map(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dominant expert ids ``[B, h, w]`` and their histogram ``[E]``."""
    if probs.ndim != 4:
        raise ValueError(f"token probs must be [B, E, h, w], got {tuple(probs.shape)}")
    ids = probs.argmax(dim=1)
    hist = torch.bincount(ids.reshape(-1), minlength=probs.shape[1])
    return ids, hist


def overlay_token_heatmap(
    image: np.ndarray,
    probs: torch.Tensor,
    *,
    alpha: float = DEFAULT_ALPHA,
    cmap: str = DEFAULT_CMAP,
) -> np.ndarray:
    """Alpha-blend the dominant-expert map, resized to the image, onto ``image``."""
    from matplotlib import colormaps

    height, width = image.shape[:2]
    resized = nn.functional.interpolate(
        probs.float(), size=(height, width), mode="bilinear", align_corners=False
    )
    dominant = resized.argmax(dim=1)[0]
    experts = int(probs.shape[1])
    scaled = (dominant.float() / max(experts - 1, 1)).numpy()
    colors = colormaps[cmap](scaled)[..., :3]
    base = image.astype(np.float32)
    if image.dtype == np.uint8:
        base = base / 255.0
    blended = (1.0 - alpha) * base[..., :3] + alpha * colors
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def render_token_figure(
    image: np.ndarray,
    panels: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    title: str,
    cmap: str = DEFAULT_CMAP,
) -> None:
    """Write one figure: input image plus one alpha-overlay panel per layer.

    Dark theme; color encodes the dominant-expert id (0..E-1), disclosed via a
    colorbar so the reader never mistakes it for a routing-weight magnitude.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    for _font in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        try:
            font_manager.fontManager.addfont(_font)
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    bg, fg, sub, mut, line = "#0d1117", "#f0f6fc", "#c9d1d9", "#8b949e", "#30363d"
    ncols = 1 + len(panels)
    figure, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.9))
    figure.patch.set_facecolor(bg)
    axes = np.atleast_1d(axes)
    axes[0].imshow(image)
    axes[0].set_title("原图", fontsize=12, color=fg)
    for axis, panel in zip(axes[1:], panels):
        axis.imshow(panel["overlay"])
        axis.set_title(str(panel["label"]), fontsize=12, color=fg)
        axis.set_xlabel(str(panel["note"]), fontsize=9, color=sub)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor(bg)
        for spine in axis.spines.values():
            spine.set_color(line)

    figure.suptitle(title, fontsize=13, color=fg, y=0.97)
    figure.text(
        0.02,
        0.02,
        "图为 local_conv 空间分支 softmax（上游每图 top-k 决策另见 provenance）",
        fontsize=8,
        color=mut,
    )
    figure.subplots_adjust(left=0.02, right=0.90, top=0.86, bottom=0.13, wspace=0.08)

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar_ax = figure.add_axes([0.915, 0.13, 0.02, 0.68])
    cbar = figure.colorbar(sm, cax=cbar_ax)
    cbar.set_label("主导专家编号\n0 → E−1", fontsize=8, color=sub)
    cbar.ax.tick_params(colors=mut, labelsize=8)
    cbar.outline.set_edgecolor(line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=bg)
    plt.close(figure)


class RouterTokenProbe:
    """Capture one routing module's probabilities through forward hooks only."""

    def __init__(self, router: nn.Module, name: str) -> None:
        self.name = name
        self.router_probs: torch.Tensor | None = None
        self.spatial_logits: torch.Tensor | None = None
        self._handles = [router.register_forward_hook(self._on_router)]
        spatial = getattr(router, "local_conv", None)
        if isinstance(spatial, nn.Module):
            self._handles.append(spatial.register_forward_hook(self._on_spatial))

    def _on_router(self, module: nn.Module, args: Any, output: Any) -> None:
        self.router_probs = extract_router_probs(output)

    def _on_spatial(self, module: nn.Module, args: Any, output: Any) -> None:
        self.spatial_logits = output if isinstance(output, torch.Tensor) else None

    def reset(self) -> None:
        self.router_probs = None
        self.spatial_logits = None

    def token_probs(self) -> tuple[torch.Tensor, str]:
        return resolve_token_probs(self.router_probs, self.spatial_logits)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _load_val_inputs(yolo: Any, config: Mapping[str, Any]) -> list[tuple[str, torch.Tensor]]:
    """coco8 val tensors (letterboxed, /255) paired with their source image path.

    Mirrors ``scripts/run_e3_smoke.py::_moe_val_sample_tensors`` (same validator
    pipeline, batch=1, rect) so the overlay input is the P0/P1 MoE evidence
    input; it additionally keeps ``batch["im_file"]`` for the overlay.
    """
    import ultralytics
    from ultralytics.data.utils import check_det_dataset

    from scripts.run_e3_smoke import _require_moe_validator_api

    moe = config["moe"]
    device = str(config.get("device", "cpu"))
    args = {
        **getattr(yolo, "overrides", {}),
        "rect": True,
        "data": moe.get("dataset", "coco8.yaml"),
        "split": moe.get("dataset_split", "val"),
        "batch": int(moe.get("batch", 1)),
        "device": device,
        "verbose": False,
        "mode": "val",
    }
    validator = _require_moe_validator_api(yolo, str(ultralytics.__version__))(args=args, _callbacks={})
    validator.training = False
    validator.args.workers = 0
    validator.args.rect = True
    validator.data = check_det_dataset(validator.args.data, split=validator.args.split)
    validator.stride = int(yolo.model.stride.max().item())
    validator.device = torch.device(device)
    loader = validator.get_dataloader(validator.data.get(validator.args.split), validator.args.batch)
    return [(str(batch["im_file"][0]), validator.preprocess(batch)["img"]) for batch in loader]


def render_token_heatmaps(
    config: Mapping[str, Any],
    baseline_root: Path,
    out_dir: Path,
    *,
    alpha: float = DEFAULT_ALPHA,
    cmap: str = DEFAULT_CMAP,
) -> dict[str, Any]:
    """Run one train-mode forward per coco8 val image and render the overlays."""
    from scripts.routing_capture import (
        module_family,
        restore_bn_running_state,
        snapshot_bn_running_state,
    )

    os.chdir(baseline_root)
    from ultralytics import YOLO

    yolo = YOLO(str(config["moe"]["model_config"]))
    model = yolo.model.train()  # the router publishes router_probs in train mode only
    probes: dict[str, RouterTokenProbe] = {}
    for name, module in model.named_modules():
        if module_family(module) != "moe":
            continue
        router = getattr(module, "routing", None)
        if isinstance(router, nn.Module):
            probes[name] = RouterTokenProbe(router, name)
    if not probes:
        raise RuntimeError("no MoE routing module found in the baseline model")

    inputs = _load_val_inputs(yolo, config)
    bn_state = snapshot_bn_running_state(model)
    per_image: list[dict[str, Any]] = []
    try:
        for im_file, tensor in inputs:
            for probe in probes.values():
                probe.reset()
            model(tensor)
            layers: list[dict[str, Any]] = []
            for name, probe in probes.items():
                probs, source = probe.token_probs()
                ids, histogram = dominant_expert_map(probs)
                published = probe.router_probs
                published_ids = published.argmax(dim=1)
                layers.append(
                    {
                        "layer": name,
                        "num_experts": int(probs.shape[1]),
                        "token_source": source,
                        "token_probs_shape": list(probs.shape),
                        "published_router_probs_shape": list(published.shape),
                        "published_dominant_expert": int(published_ids[0]),
                        "published_dominant_share": float(published[0, published_ids[0]]),
                        "token_dominant_expert": int(histogram.argmax()),
                        "token_expert_histogram": [int(value) for value in histogram.tolist()],
                        "token_probs": probs,
                        "token_ids": ids,
                    }
                )
            per_image.append({"im_file": im_file, "input_shape": list(tensor.shape), "layers": layers})
    finally:
        restore_bn_running_state(model, bn_state)
        for probe in probes.values():
            probe.close()

    import matplotlib.image as mpimg

    out_dir = Path(out_dir)
    figures: list[dict[str, Any]] = []
    for record in per_image:
        image = mpimg.imread(record["im_file"])
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        panels: list[dict[str, Any]] = []
        for layer in record["layers"]:
            panels.append(
                {
                    "overlay": overlay_token_heatmap(image, layer["token_probs"], alpha=alpha, cmap=cmap),
                    "label": f"{layer['layer']} ｜ E={layer['num_experts']}",
                    "note": (
                        f"token 主导 e{layer['token_dominant_expert']} ｜ "
                        f"每图决策 e{layer['published_dominant_expert']} "
                        f"p={layer['published_dominant_share']:.2f}"
                    ),
                }
            )
        name = Path(record["im_file"]).name
        png = out_dir / f"token_heatmap_{name}.png"
        render_token_figure(
            image,
            panels,
            png,
            title="MoE 路由热图（颜色 = 主导专家编号）",
            cmap=cmap,
        )
        figures.append(
            {
                "png": png.name,
                "image": name,
                "image_sha256": _sha256(Path(record["im_file"])),
                "input_tensor_shape": record["input_shape"],
                "layers": [
                    {key: value for key, value in layer.items() if key not in NON_TENSOR_KEYS}
                    for layer in record["layers"]
                ],
            }
        )
        logger.info("rendered %s (%d layers)", png.name, len(panels))

    metadata = {
        "figure_version": FIGURE_VERSION,
        "generator": GENERATOR,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_config": str(config["moe"]["model_config"]),
        "dataset": f"{config['moe'].get('dataset')}:{config['moe'].get('dataset_split', 'val')}",
        "device": str(config.get("device", "cpu")),
        "alpha": alpha,
        "cmap": cmap,
        "router_probs_note": (
            "the upstream DualStreamGateRouter publishes router_probs only in train mode and "
            "per-image [B, E]; the per-token map is the softmax over its captured spatial "
            "branch (local_conv) logits"
        ),
        "figures": figures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # artifacts/** is hash-pinned with LF endings (.gitattributes: -text), so the
    # provenance file must not pick up the platform newline translation.
    with (out_dir / "token_heatmap_provenance.json").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Render MoE token-routing heatmaps overlaid on the coco8 val images"
    )
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "e3_smoke.yaml"))
    parser.add_argument("--baseline-root", default=None, help="Deployed YOLO-Master checkout to chdir into")
    parser.add_argument("--out-dir", default="artifacts/figures/p2", help="Output directory for the PNGs")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Overlay alpha")
    parser.add_argument("--cmap", default=DEFAULT_CMAP, help="Matplotlib colormap for expert ids")
    args = parser.parse_args(argv)

    from scripts.run_e3_smoke import apply_seed, load_config, resolve_baseline_root

    config = load_config(Path(args.config))
    apply_seed(config)
    baseline_root = resolve_baseline_root(config, args.baseline_root)
    if not baseline_root.is_dir():
        raise SystemExit(f"baseline_root not found: {baseline_root}")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PACKAGE_ROOT / out_dir
    metadata = render_token_heatmaps(config, baseline_root, out_dir, alpha=args.alpha, cmap=args.cmap)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CMAP",
    "FIGURE_VERSION",
    "GENERATOR",
    "RouterTokenProbe",
    "dominant_expert_map",
    "extract_router_probs",
    "overlay_token_heatmap",
    "render_token_figure",
    "render_token_heatmaps",
    "resolve_token_probs",
]


if __name__ == "__main__":
    raise SystemExit(main())
