#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — GCFC (Gaussian support field correction) tests, manuscript §2.3.5.

  T1  H_i is a smooth, max-normalised field whose peak sits on the track points
  T2  G_i = <H_i, M_i> / (||H_i||_1 + eps) is in [0, 1] and rises when the
      tracks fall inside the mask
  T3  Boundary band B_i contains the mask contour only (no interior, no far
      exterior)
  T4  Gaussian backend expands ONLY inside B_i, never into the mask interior or
      far outside
  T5  Support gate blocks the correction when G_i < geometry_min_support_ratio
  T6  Negative deletion stays OFF for the Gaussian backend even when requested
  T7  Legacy `colmap_tracks` backend is unchanged by the GCFC additions
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/"
                "S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py")
MOD_NAME = "生成RAP-FSAM3掩码_phase21_gcfc"


def load_pipeline_module():
    if MOD_NAME in sys.modules:
        return sys.modules[MOD_NAME]
    spec = importlib.util.spec_from_file_location(MOD_NAME, PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MOD_NAME] = mod
    script_dir = str(PIPELINE.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return load_pipeline_module()


def make_args(mod, **overrides):
    """Full production defaults, then the GCFC overrides."""
    saved = sys.argv
    sys.argv = ["x", "--input_dir", "/tmp/gcfc_in", "--output_dir", "/tmp/gcfc_out"]
    try:
        args = mod.parse_args()
    finally:
        sys.argv = saved
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_observation(mod, points):
    return mod.ColmapObservation(
        image_name="frame.png", mask_stem="frame", points=np.asarray(points, dtype=np.float64))


def square_mask(size=64, lo=20, hi=44):
    mask = np.zeros((size, size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask


def contour_points(mask, step=2):
    """Points on the mask contour, in (x, y) order as COLMAP projections are."""
    ys, xs = np.nonzero(mask)
    pts = []
    for y, x in zip(ys, xs):
        # keep only pixels adjacent to the outside
        window = mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        if window.all():
            continue
        pts.append((x, y))
    return np.asarray(pts[::step], dtype=np.float64)


# ── T1 ───────────────────────────────────────────────────────────────────────
def test_t1_support_field_shape(mod):
    pts = np.array([[32.0, 32.0]])
    field = mod.points_to_support_field(pts, (64, 64), sigma=3.0)
    assert field.shape == (64, 64)
    assert field.max() == pytest.approx(1.0, abs=1e-6)
    assert field[32, 32] == pytest.approx(1.0, abs=1e-6)
    # smooth: strictly between 0 and 1 at ~1 sigma away
    assert 0.0 < field[32, 35] < 1.0
    # compact: negligible at 5 sigma
    assert field[32, 52] < 1e-3
    # empty input -> empty field, not a crash
    empty = mod.points_to_support_field(np.zeros((0, 2)), (64, 64), sigma=3.0)
    assert empty.sum() == 0.0


# ── T2 ───────────────────────────────────────────────────────────────────────
def test_t2_support_ratio(mod):
    mask = square_mask()
    inside_pts = np.argwhere(mask)[:, ::-1].astype(np.float64)  # (x, y)
    grid = np.stack(np.meshgrid(np.arange(2, 16), np.arange(2, 16)), axis=-1)
    outside_pts = grid.reshape(-1, 2).astype(np.float64)  # far from the square
    field_in = mod.points_to_support_field(inside_pts, mask.shape, 3.0)
    field_out = mod.points_to_support_field(outside_pts, mask.shape, 3.0)
    g_in = float((field_in * mask).sum() / max(field_in.sum(), 1e-6))
    g_out = float((field_out * mask).sum() / max(field_out.sum(), 1e-6))
    assert 0.0 <= g_in <= 1.0 + 1e-6
    assert 0.0 <= g_out <= 1.0 + 1e-6
    assert g_in > g_out
    # filled interior: the Gaussian tail leaks a little outside the square, so
    # G_i sits high but not at exactly 1
    assert g_in > 0.7
    assert g_out < 0.05


# ── T3 ───────────────────────────────────────────────────────────────────────
def test_t3_boundary_band(mod):
    mask = square_mask()
    band = mod.boundary_band(mask, 3)
    assert band.any()
    # contour pixels belong to the band
    assert band[20, 20]
    assert band[20, 32]
    # interior (more than 3 px from the contour) does not
    assert not band[32, 32]
    # far exterior does not
    assert not band[5, 5]
    # radius 0 disables the band
    assert mod.boundary_band(mask, 0).sum() == 0


# ── T4 ───────────────────────────────────────────────────────────────────────
def test_t4_positive_expansion_only_in_band(mod):
    args = make_args(
        mod,
        use_corrective_geometry=True,
        corrective_geometry_backend="colmap_tracks_gaussian",
        geometry_gaussian_sigma=2.0,
        geometry_support_threshold=0.2,
        geometry_boundary_band_px=6,
        geometry_min_support_ratio=0.3,
        geometry_positive_min_points=1,
        geometry_enable_negative_correction=False,
        # this test measures the GCFC expansion semantics, not the delta guard,
        # so the guard is loosened explicitly rather than bypassed silently
        geometry_correct_max_delta_ratio=0.30,
        geometry_correct_min_iou=0.50,
        fill_holes=False,
        keep_largest_component=False,
    )
    # mask misses the left part of the plant; tracks cover the true contour
    truth = np.zeros((64, 64), dtype=bool)
    truth[16:48, 16:48] = True
    base = truth.copy()
    base[:, 16:19] = False
    pts = contour_points(truth, step=1)
    final, info = mod.apply_corrective_geometry(base, "frame.png",
                                                make_observation(mod, pts), args)
    added = final & ~base
    assert info["后端"] == "colmap_tracks_gaussian"
    assert 0.0 < info["支撑场比例G"] <= 1.0
    assert added.any(), "tracks on the true contour should add the missing boundary"
    band = mod.boundary_band(base, 6)
    assert not (added & ~band).any(), "GCFC added pixels outside the boundary band"
    # far background untouched
    assert not final[:8, :].any()
    assert not final[56:, :].any()
    assert not final[:, 56:].any()


# ── T5 ───────────────────────────────────────────────────────────────────────
def test_t5_support_gate(mod):
    args = make_args(
        mod,
        use_corrective_geometry=True,
        corrective_geometry_backend="colmap_tracks_gaussian",
        geometry_gaussian_sigma=2.0,
        geometry_min_support_ratio=0.99,
        geometry_boundary_band_px=8,
        geometry_positive_min_points=1,
        geometry_enable_negative_correction=False,
        fill_holes=False,
        keep_largest_component=False,
    )
    base = square_mask()
    # tracks far from the mask -> G_i ~ 0 -> gate must reject
    pts = np.array([[4.0, 4.0], [6.0, 4.0], [4.0, 6.0]], dtype=np.float64)
    final, info = mod.apply_corrective_geometry(base, "frame.png",
                                                make_observation(mod, pts), args)
    assert info["原因"] in ("support_gate", "no_corrective_delta")
    assert info["支撑场比例G"] < 0.99
    assert np.array_equal(final, base), "gated correction must return the input mask"


# ── T6 ───────────────────────────────────────────────────────────────────────
def test_t6_negative_deletion_off(mod):
    args = make_args(
        mod,
        use_corrective_geometry=True,
        corrective_geometry_backend="colmap_tracks_gaussian",
        geometry_gaussian_sigma=3.0,
        geometry_boundary_band_px=6,
        geometry_min_support_ratio=0.0,
        geometry_positive_min_points=1,
        geometry_enable_negative_correction=True,   # requested...
        fill_holes=False,
        keep_largest_component=False,
    )
    base = square_mask()
    pts = contour_points(base, step=1)
    final, info = mod.apply_corrective_geometry(base, "frame.png",
                                                make_observation(mod, pts), args)
    # ...but GCFC has no negative branch: nothing may be removed
    assert not (base & ~final).any(), "GCFC must not delete foreground pixels"
    assert info["负修正删除像素"] == 0


# ── T7 ───────────────────────────────────────────────────────────────────────
def test_t7_legacy_backend_unchanged(mod):
    args = make_args(
        mod,
        use_corrective_geometry=True,
        corrective_geometry_backend="colmap_tracks",
        geometry_track_dilation=7,
        geometry_positive_min_points=1,
        geometry_enable_negative_correction=False,
        fill_holes=False,
        keep_largest_component=False,
    )
    base = square_mask()
    pts = np.array([[52.0, 32.0]], dtype=np.float64)  # just right of the mask
    final, info = mod.apply_corrective_geometry(base, "frame.png",
                                                make_observation(mod, pts), args)
    assert info["后端"] == "colmap_tracks"
    assert info["支撑场比例G"] == -1.0, "legacy backend must not compute G_i"
    assert final[32, 46:50].any(), "legacy dilation support should still expand"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
