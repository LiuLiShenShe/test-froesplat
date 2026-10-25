#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — M2M3-GO unit tests (CPU only).

Covers the manuscript primitives implemented in
`2d-gaussian-splatting-m2m3-floor40/utils/m2m3_go.py`:

  T1 projection convention / in-view mask
  T2 footprint mean on a constant field
  T3 boundary distance + support maps
  T4 utility monotonicity in S_m2g, S_g2m, boundary support and residual
  T5 local Floor40 blocks and floor blocking
  T6 local block assignment range
  T7 deficit Gaussian attribute shapes and SH-DC round-trip
  T8 deficit pixel selection (budget + stride)
  T9 back-projection of the principal point
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

TRAINER = Path("/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40")
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from utils import m2m3_go as mg  # noqa: E402


class FakeCam:
    def __init__(self, proj=None, width=64, height=48, fovx=1.0, fovy=0.8, R=None, T=None):
        self.full_proj_transform = torch.eye(4) if proj is None else proj
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        self.R = np.eye(3) if R is None else R
        self.T = np.zeros(3) if T is None else T


def test_t1_projection_and_in_view():
    cam = FakeCam(proj=torch.eye(4))
    xyz = torch.tensor([[0.5, 0.5, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    pix, depth, in_view = mg.project_centers(xyz, cam)
    assert torch.allclose(depth, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(pix[0], torch.tensor([0.5, 0.5]))
    assert in_view.tolist() == [True, False, True]


def test_t2_footprint_mean_constant_field():
    field = torch.ones(16, 16)
    pix = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    radii = torch.tensor([1.0, 5.0])
    in_view = torch.tensor([True, True])
    mean = mg.footprint_mean(field, pix, radii, in_view)
    assert torch.allclose(mean, torch.ones(2), atol=1e-5)


def test_t3_boundary_maps():
    mask = torch.zeros(1, 21, 21)
    mask[:, 6:15, 6:15] = 1.0
    distance = mg.boundary_distance_map(mask)
    assert distance.shape == mask.shape
    assert distance[0, 10, 10] > distance[0, 6, 10]
    support = mg.boundary_support_map(mask, sigma=1.0)
    assert support.max() <= 1.0 + 1e-6
    assert support[0, 6, 10] > support[0, 10, 10]


def test_t4_utility_monotonicity():
    n = 4
    base = dict(
        s_m2g=torch.tensor([0.1, 0.9, 0.5, 0.5]),
        s_g2m=torch.tensor([0.1, 0.1, 0.9, 0.1]),
        b_support=torch.tensor([0.1, 0.1, 0.1, 0.9]),
        opacity=torch.full((n,), 0.8),
        visible_views=torch.full((n,), 4.0),
        residual=torch.tensor([0.01, 0.01, 0.01, 0.01]),
    )
    utility = mg.utility_score(**base, weights=(1.0, 0.5, 0.5, 0.5, 0.25, 0.5))
    assert utility[1] > utility[0], "higher S_m2g must raise utility"
    assert utility[2] > utility[0], "higher S_g2m must raise utility"
    assert utility[3] > utility[0], "higher boundary support must raise utility"
    # residual term
    base2 = dict(base)
    base2["residual"] = torch.tensor([0.01, 0.01, 0.01, 0.01])
    u_low = mg.utility_score(**base2, weights=(1.0, 0.5, 0.5, 0.5, 0.25, 0.5))
    base2["residual"] = torch.tensor([0.9, 0.9, 0.9, 0.9])
    u_high = mg.utility_score(**base2, weights=(1.0, 0.5, 0.5, 0.5, 0.25, 0.5))
    assert torch.all(u_high >= u_low)


def test_t5_local_floor_blocks_removals():
    utility = torch.tensor([0.1, 0.2, 0.3, 0.4])
    block_ids = torch.tensor([0, 0, 1, 1])
    counts = torch.tensor([2, 2])
    minima = torch.tensor([1, 1])
    remove, report = mg.local_floor_select(
        torch.arange(4), utility, block_ids, counts, minima, max_remove=3)
    assert remove.sum().item() == 2
    assert report["floor_blocked"] == 2
    assert remove.tolist() == [True, False, True, False]


def test_t5b_floor_allows_when_above_minimum():
    utility = torch.tensor([0.1, 0.2, 0.3, 0.4])
    block_ids = torch.tensor([0, 0, 0, 0])
    counts = torch.tensor([4])
    minima = torch.tensor([1])
    remove, report = mg.local_floor_select(
        torch.arange(4), utility, block_ids, counts, minima, max_remove=3)
    assert remove.sum().item() == 3
    assert report["floor_blocked"] == 0


def test_t6_local_block_assignment():
    xyz = torch.rand(100, 3) * 2.0 - 1.0
    blocks = mg.assign_local_blocks(xyz, n_blocks_per_axis=4)
    assert blocks.min() >= 0 and blocks.max() < 4 ** 3
    assert blocks.shape == (100,)


def test_t7_deficit_gaussian_attributes():
    points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    colors = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]])
    attrs = mg.make_deficit_gaussians(points, colors, base_scale=0.01, sh_degree=3)
    # This fork's GaussianModel stores features as
    #   _features_dc   = features[:, :, 0:1].transpose(1, 2)  -> (N, 1, 3)
    #   _features_rest = features[:, :, 1:].transpose(1, 2)   -> (N, (d+1)^2-1, 3)
    assert attrs["xyz"].shape == (2, 3)
    assert attrs["f_dc"].shape == (2, 1, 3)
    assert attrs["f_rest"].shape == (2, 15, 3)
    assert attrs["opacity"].shape == (2, 1)
    # 2DGS surfels carry two in-plane scale axes
    assert attrs["scaling"].shape == (2, 2)
    assert attrs["rotation"].shape == (2, 4)
    decoded = torch.clamp(attrs["f_dc"].squeeze(1) * 0.28209479177387814 + 0.5, 0, 1)
    assert torch.allclose(decoded, colors, atol=1e-5)
    # sh_degree drives the f_rest width, matching GaussianModel.extra_f_names
    for degree in (0, 1, 2):
        attrs_d = mg.make_deficit_gaussians(points, colors, base_scale=0.01, sh_degree=degree)
        assert attrs_d["f_rest"].shape == (2, (degree + 1) ** 2 - 1, 3)


def test_t8_select_deficit_pixels():
    mask = torch.ones(1, 20, 20)
    alpha = torch.zeros(1, 20, 20)
    depth = torch.ones(1, 20, 20)
    pixels, depths = mg.select_deficit_pixels(mask, alpha, depth, threshold=0.5,
                                              budget=10, grid_stride=1)
    assert pixels.shape[0] == 10
    assert depths.shape[0] == 10
    assert torch.allclose(depths, torch.ones(10))
    pixels_coarse, _ = mg.select_deficit_pixels(mask, alpha, depth, threshold=0.5,
                                                budget=1000, grid_stride=4)
    assert pixels_coarse.shape[0] <= 25


def test_t9_backproject_principal_point():
    cam = FakeCam(width=64, height=48, fovx=1.0, fovy=0.8, R=np.eye(3), T=np.zeros(3))
    pixels = torch.tensor([[32.0, 24.0]])
    depths = torch.tensor([2.0])
    points = mg.backproject_pixels(pixels, depths, cam)
    assert points.shape == (1, 3)
    assert torch.allclose(points[0], torch.tensor([0.0, 0.0, 2.0]), atol=1e-5)


def test_t6b_tau_p_resolution_modes():
    utility = torch.linspace(0.0, 1.0, 101)
    valid = torch.ones(101, dtype=torch.bool)
    fixed, source = mg.resolve_tau_p(utility, valid, "fixed", 0.15)
    assert fixed == 0.15 and source == "fixed"
    # quantile mode must return a value that actually separates the distribution
    q, source = mg.resolve_tau_p(utility, valid, "quantile", 0.15, quantile=0.01)
    assert source.startswith("quantile")
    assert 0.0 <= q <= 0.15
    assert int((utility < q).sum()) == 1
    # an all-invalid round falls back to the fixed hyperparameter
    q_empty, source_empty = mg.resolve_tau_p(utility, torch.zeros(101, dtype=torch.bool),
                                             "quantile", 0.15, quantile=0.01)
    assert q_empty == 0.15 and source_empty == "fixed(empty)"


def test_t10_select_densify_primitives():
    """Densify(g_j) = 1[S_m2g>tau_m] * 1[S_g2m>tau_h or R>tau_r]."""
    s_m2g = torch.tensor([0.90, 0.90, 0.90, 0.10, 0.90])
    s_g2m = torch.tensor([0.50, 0.10, 0.10, 0.90, 0.05])
    residual = torch.tensor([0.00, 0.00, 0.00, 0.00, 0.50])
    valid = torch.ones(5, dtype=torch.bool)
    selected = mg.select_densify_primitives(
        s_m2g, s_g2m, residual, valid, tau_m=0.65, tau_h=0.35, tau_r=0.12, budget=8)
    # 1,2 fail both deficit terms; 3 has weak mask support; 4 qualifies via R
    assert sorted(selected.tolist()) == [0, 4]
    # ranking is by deficit evidence (S_g2m + R): primitive 4 scores 0.55 > 0.50
    top1 = mg.select_densify_primitives(
        s_m2g, s_g2m, residual, valid, tau_m=0.65, tau_h=0.35, tau_r=0.12, budget=1)
    assert top1.tolist() == [4]
    none = mg.select_densify_primitives(
        s_m2g, s_g2m, residual, torch.zeros(5, dtype=torch.bool),
        tau_m=0.65, tau_h=0.35, tau_r=0.12, budget=8)
    assert none.numel() == 0


def test_t11_split_attribute_tensors():
    n = 4
    xyz = torch.zeros(n, 3)
    f_dc = torch.zeros(n, 1, 3)
    f_rest = torch.zeros(n, 15, 3)
    opacity = torch.full((n, 1), 2.0)
    scaling = torch.zeros(n, 2)
    rotation = torch.zeros(n, 4)
    rotation[:, 0] = 1.0
    index = torch.tensor([1, 3])
    out = mg.split_attribute_tensors(
        xyz, f_dc, f_rest, opacity, scaling, rotation, index,
        scale_shrink=1.0, jitter_ratio=0.5, generator=torch.Generator().manual_seed(0))
    assert out["xyz"].shape == (2, 3)
    assert out["f_dc"].shape == (2, 1, 3)
    assert out["f_rest"].shape == (2, 15, 3)
    assert out["scaling"].shape == (2, 2)
    assert out["parent_index"].tolist() == [1, 3]
    # identity rotation: the jitter stays inside the surfel plane (z = 0)
    assert torch.allclose(out["xyz"][:, 2], torch.zeros(2), atol=1e-6)
    # the offset is jitter_ratio * scale * N(0,1); 4 sigma bounds the seed-0 draw
    assert out["xyz"].abs().max() <= 4.0 * 0.5
    # empty selection must stay shape-compatible with the optimiser
    empty = mg.split_attribute_tensors(xyz, f_dc, f_rest, opacity, scaling, rotation,
                                       index[:0])
    assert empty["xyz"].shape == (0, 3) and empty["f_rest"].shape == (0, 15, 3)


def test_t12_remap_surviving_indices():
    """Densify parents are scored pre-prune but split post-prune."""
    remove_mask = torch.tensor([False, True, False, True, False])
    # surviving primitives are old 0, 2, 4 -> new 0, 1, 2
    selected = torch.tensor([0, 1, 2, 3, 4])
    mapped = mg.remap_surviving_indices(selected, remove_mask)
    assert mapped.tolist() == [0, 1, 2]
    # a parent pruned in the same round is dropped, never mapped onto a neighbour
    only_pruned = mg.remap_surviving_indices(torch.tensor([1, 3]), remove_mask)
    assert only_pruned.numel() == 0
    # no removals leaves the indices untouched
    untouched = mg.remap_surviving_indices(selected, torch.zeros(5, dtype=torch.bool))
    assert untouched.tolist() == [0, 1, 2, 3, 4]
    # mapping must agree with the actual compaction of a tensor
    values = torch.arange(5) * 10.0
    assert torch.equal(values[~remove_mask][mapped], values[torch.tensor([0, 2, 4])])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
