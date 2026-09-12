"""Tests for the P2 MoE token-routing heatmap renderer (no baseline model needed).

Covers the four surfaces the P2 slice pins down: the forward-hook output shape,
per-token probability resolution (rank-4 as-is, rank-2 + spatial branch), the
heatmap/overlay image sizes, and the error when no router_probs is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.render_token_heatmap import (
    RouterTokenProbe,
    dominant_expert_map,
    extract_router_probs,
    overlay_token_heatmap,
    render_token_figure,
    resolve_token_probs,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _FakeRouter(torch.nn.Module):
    """Minimal stand-in for an upstream dual-stream router (same output tuple)."""

    def __init__(self, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.local_conv = torch.nn.Conv2d(3, num_experts, 1)

    def forward(self, x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x = torch.randn(1, 3, 8, 8) if x is None else x
        logits = self.local_conv(x)  # [B, E, h, w]
        probs = torch.softmax(logits.mean(dim=[2, 3]), dim=1)  # [B, E] per image
        indices = torch.topk(probs, self.top_k, dim=1).indices
        weights = probs.unsqueeze(-1).unsqueeze(-1)
        return weights, indices, {"router_probs": probs, "topk_indices": indices}


def test_probe_captures_hook_output_and_resolves_token_probs() -> None:
    router = _FakeRouter(num_experts=4)
    probe = RouterTokenProbe(router, "model.5")
    try:
        probe.reset()
        router()
        assert probe.router_probs is not None
        assert tuple(probe.router_probs.shape) == (1, 4)
        assert probe.spatial_logits is not None
        assert tuple(probe.spatial_logits.shape) == (1, 4, 8, 8)
        probs, source = probe.token_probs()
        assert tuple(probs.shape) == (1, 4, 8, 8)
        assert source == "router_spatial_branch_softmax"
        assert torch.allclose(probs.sum(dim=1), torch.ones(1, 8, 8), atol=1e-5)
    finally:
        probe.close()


def test_extract_router_probs_accepts_tuple_and_mapping() -> None:
    probs = torch.rand(2, 3, 4, 4)
    stats = {"router_probs": probs}
    assert tuple(extract_router_probs((torch.rand(2, 3, 1, 1), torch.zeros(2, 3), stats)).shape) == (2, 3, 4, 4)
    assert tuple(extract_router_probs(stats).shape) == (2, 3, 4, 4)


def test_extract_router_probs_raises_without_router_probs() -> None:
    with pytest.raises(ValueError, match="router_probs"):
        extract_router_probs((torch.rand(1, 2, 1, 1), torch.zeros(1, 2), {"topk_indices": torch.zeros(1, 2)}))
    with pytest.raises(ValueError, match="no routing-stats mapping"):
        extract_router_probs(torch.rand(1, 2, 1, 1))


def test_resolve_uses_rank4_router_probs_directly() -> None:
    probs = torch.rand(1, 5, 3, 4)
    resolved, source = resolve_token_probs(probs, torch.rand(1, 5, 9, 9))
    assert source == "router_probs"
    assert resolved is probs


def test_resolve_rejects_per_image_probs_without_spatial_branch() -> None:
    with pytest.raises(ValueError, match="per-image"):
        resolve_token_probs(torch.rand(1, 4))
    with pytest.raises(ValueError, match="no router_probs captured"):
        resolve_token_probs(None)


def test_dominant_expert_map_shape_and_histogram() -> None:
    probs = torch.softmax(torch.randn(1, 3, 4, 5), dim=1)
    ids, histogram = dominant_expert_map(probs)
    assert tuple(ids.shape) == (1, 4, 5)
    assert int(histogram.sum()) == 20
    assert len(histogram) == 3


def test_overlay_keeps_the_image_size_and_dtype() -> None:
    image = (np.random.default_rng(0).random((12, 16, 3)) * 255).astype(np.uint8)
    probs = torch.softmax(torch.randn(1, 4, 3, 2), dim=1)
    overlay = overlay_token_heatmap(image, probs, alpha=0.5, cmap="turbo")
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_render_token_figure_writes_png(tmp_path: Path) -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    panel = {
        "overlay": overlay_token_heatmap(image, torch.softmax(torch.randn(1, 2, 4, 4), dim=1)),
        "label": "model.5 | E=2",
        "note": "published per-image e0 p=0.50 | token top e0",
    }
    out = tmp_path / "token_heatmap.png"
    render_token_figure(image, [panel], out, title="unit test")
    assert out.read_bytes()[:8] == PNG_MAGIC
