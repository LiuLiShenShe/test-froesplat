"""Unit tests for the multi-view fusion module (no GPU / weights needed)."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresplat_adapter.fusion import (  # noqa: E402
    FusionConfig,
    build_observation_table,
    depth_edge_mask,
    estimate_view_normals,
    export_observations_csv,
    fuse_views,
    load_observations_npz,
    normal_bin_ids,
    pixel_index_to_xy,
    save_observations_npz,
)
from foresplat_adapter.fusion.adaptive_voxel import choose_voxel_size, voxel_keys  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _plane_view(
    z: float,
    normal: np.ndarray,
    camera_center: np.ndarray,
    size: int = 32,
    jitter: float = 0.0,
    conf: float = 2.0,
    image_id: int = 1,
    seed: int = 0,
) -> dict:
    """A synthetic view of a flat patch at height ``z`` (facing ``normal``)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1.0, 1.0, size)
    ys = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z)
    points = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
    if jitter:
        points += rng.normal(0.0, jitter, points.shape).astype(np.float32)

    h, w, _ = points.shape
    valid = np.ones((h, w), dtype=bool)
    camera_center = np.asarray(camera_center, dtype=np.float32)

    # Depth for the edge mask: distance along the camera ray.
    depth = np.linalg.norm(points - camera_center[None, None, :], axis=-1)

    normals_data = estimate_view_normals(points, valid, camera_center, depth=depth)
    true_normal = np.asarray(normal, dtype=np.float32)
    normals = np.broadcast_to(true_normal, points.shape).copy()

    colors = np.zeros((h, w, 3), dtype=np.float32)
    colors[..., 0] = 0.5
    conf_map = np.full((h, w), conf, dtype=np.float32)
    return {
        "points": points,
        "valid": valid,
        "colors": colors,
        "conf": conf_map,
        "normals": normals,          # exact normals (fusion input)
        "quality": normals_data["quality"],
        "edge": normals_data["edge"],
        "image_id": image_id,
        "depth": depth,
    }


# --------------------------------------------------------------------------
# normal estimation
# --------------------------------------------------------------------------

class TestNormalEstimation:
    def test_flat_plane_normal_and_orientation(self):
        size = 32
        xs = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(xs, xs)
        points = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).astype(np.float32)
        valid = np.ones((size, size), dtype=bool)
        camera = np.array([0.0, 0.0, 5.0])
        result = estimate_view_normals(points, valid, camera)
        # interior pixels: normal should be +/-z, oriented towards the camera -> +z
        interior = result["normals"][5:-5, 5:-5, 2]
        assert np.allclose(interior, 1.0, atol=1e-3)
        assert np.abs(result["normals"][5:-5, 5:-5, 0]).max() < 1e-3
        # flat plane -> planarity near 1
        assert result["quality"][5:-5, 5:-5].min() > 0.95

    def test_flat_plane_flipped_camera_orients_normal(self):
        size = 32
        xs = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(xs, xs)
        points = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).astype(np.float32)
        valid = np.ones((size, size), dtype=bool)
        camera = np.array([0.0, 0.0, -5.0])  # camera below the plane
        result = estimate_view_normals(points, valid, camera)
        assert np.allclose(result["normals"][5:-5, 5:-5, 2], -1.0, atol=1e-3)

    def test_depth_edge_mask_finds_discontinuity(self):
        depth = np.ones((8, 8), dtype=np.float32)
        depth[:, 4:] = 2.0
        valid = np.ones_like(depth, dtype=bool)
        edge = depth_edge_mask(depth, valid, rtol=0.03)
        assert edge[:, 3].all() and edge[:, 4].all()
        assert not edge[:, :3].any() and not edge[:, 5:].any()

    def test_neighbour_count_and_invalid_pixels(self):
        size = 8
        points = np.zeros((size, size, 3), dtype=np.float32)
        valid = np.zeros((size, size), dtype=bool)
        valid[4, 4] = True  # a single valid pixel has < min_neighbours
        result = estimate_view_normals(points, valid, np.zeros(3))
        assert result["neighbour_count"][4, 4] == 1
        assert np.all(result["normals"] == 0.0)


# --------------------------------------------------------------------------
# voxel hashing / adaptive size
# --------------------------------------------------------------------------

class TestAdaptiveVoxel:
    def test_voxel_keys_deterministic_and_injective(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(0.0, 10.0, size=(10_000, 3))
        keys_a = voxel_keys(pts, 0.1)
        keys_b = voxel_keys(pts, 0.1)
        assert np.array_equal(keys_a, keys_b)
        # points known to be in different voxels must get different keys
        far = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        assert voxel_keys(far, 0.1)[0] != voxel_keys(far, 0.1)[1]
        # negative coordinates hash distinctly
        neg = np.array([[-3.7, -2.2, -1.1], [3.7, 2.2, 1.1]])
        assert voxel_keys(neg, 0.1)[0] != voxel_keys(neg, 0.1)[1]

    def test_choose_voxel_size_hits_target(self):
        rng = np.random.default_rng(0)
        # uniform points in a box: ~1 voxel per target
        pts = rng.uniform(0.0, 100.0, size=(500_000, 3))
        choice = choose_voxel_size(pts, target_points=50_000, nn_scale=0.01)
        assert 0.9 * 50_000 <= choice["expected_fused_points"] <= 1.5 * 50_000

    def test_normal_bin_ids_separates_opposite_normals(self):
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
        bins = normal_bin_ids(normals)
        assert bins[0] != bins[1]


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

class TestFuseViews:
    def test_two_views_of_one_plane_merge_with_provenance(self):
        view_a = _plane_view(0.0, (0, 0, 1), (0, 0, 5), image_id=1, seed=1)
        view_b = _plane_view(0.0, (0, 0, 1), (2, 0, 5), image_id=2, seed=2)
        cfg = FusionConfig(target_points=64, min_points=2, max_points=10_000)
        result = fuse_views([view_a, view_b], cfg)

        n_candidates = result["stats"]["num_candidates"]
        n_fused = result["fused"]["xyz"].shape[0]
        assert n_candidates == 2 * 32 * 32
        assert n_fused < n_candidates  # fusion actually merged
        # merged positions sit on the plane
        assert np.abs(result["fused"]["xyz"][:, 2]).max() < 1e-4
        # multi-view groups exist (observation_count > 1 for some points)
        assert result["fused"]["observation_count"].max() > 1
        # provenance: every fused point has >= 1 observation, ids consistent
        obs = result["observations"]
        assert obs["fused_point_id"].size == n_candidates
        counts = np.bincount(obs["fused_point_id"], minlength=n_fused)
        assert counts.min() >= 1
        assert np.array_equal(np.sort(np.unique(obs["fused_point_id"])), np.arange(n_fused))
        # both images appear in the provenance
        assert set(np.unique(obs["source_image_id"])) == {1, 2}

    def test_thin_leaf_two_sides_are_not_merged(self):
        # two sheets 0.001 apart inside the same voxel with opposite normals
        view_a = _plane_view(0.0, (0, 0, 1), (0, 0, 5), image_id=1, size=16)
        view_b = _plane_view(0.001, (0, 0, -1), (0, 0, -5), image_id=2, size=16)
        cfg = FusionConfig(target_points=1000, min_points=2, max_points=10_000)
        result = fuse_views([view_a, view_b], cfg)
        xyz = result["fused"]["xyz"]
        # both sheets must survive: expect points near z=0 AND z=0.001
        near_top = np.abs(xyz[:, 2] - 0.001) < 1e-4
        near_bottom = np.abs(xyz[:, 2]) < 1e-4
        assert near_top.sum() >= 100 and near_bottom.sum() >= 100
        normals = result["fused"]["normals"]
        top_normals = normals[near_top]
        bottom_normals = normals[near_bottom]
        assert top_normals[:, 2].mean() < 0      # sheet facing away
        assert bottom_normals[:, 2].mean() > 0  # sheet facing up

    def test_orthogonal_surfaces_in_one_voxel_are_not_merged(self):
        # vertical wall vs horizontal floor meeting inside one voxel
        size = 16
        xs = np.linspace(-0.05, 0.05, size)
        wall = np.stack(np.meshgrid(xs, xs), axis=-1)  # unused orientation reference
        floor_view = _plane_view(0.0, (0, 0, 1), (0, 0, 5), image_id=1, size=16)
        # a vertical wall at x=0.01 spanning the same tiny area
        wall_view = _plane_view(0.0, (1, 0, 0), (5, 0, 0), image_id=2, size=16)
        # rotate the wall's geometry so it really is vertical
        wall_points = wall_view["points"].copy()                    # (H, W, 3)
        wall_points = wall_points[..., [2, 1, 0]]                   # swap x/z -> vertical plane
        wall_view["points"] = wall_points
        cfg = FusionConfig(target_points=1000, min_points=2, max_points=10_000)
        result = fuse_views([floor_view, wall_view], cfg)
        normals = result["fused"]["normals"]
        has_up = (normals[:, 2] > 0.7).sum()
        has_sideways = (np.abs(normals[:, 0]) > 0.7).sum()
        assert has_up > 20 and has_sideways > 20
        del wall

    def test_edge_pixels_keep_higher_density(self):
        size = 32
        view_a = _plane_view(0.0, (0, 0, 1), (0, 0, 5), image_id=1, size=size)
        view_b = _plane_view(0.0, (0, 0, 1), (1, 0, 5), image_id=2, size=size)
        # mark a line of pixels in view_a as depth-discontinuity
        view_a["edge"][:, 16] = True
        cfg = FusionConfig(
            target_points=200, min_points=2, max_points=100_000,
            edge_voxel_divisor=4.0, edge_weight_boost=2.0,
        )
        result = fuse_views([view_a, view_b], cfg)
        fused = result["fused"]
        assert fused["is_edge"].sum() > 0
        # edge representatives must exist at higher density: the marked column
        # sits at x = xs[16] on the plane; keep ~size points along it
        line_points = fused["xyz"][fused["is_edge"]]
        xs = np.linspace(-1, 1, size)
        x_line = xs[16]
        on_line = line_points[np.abs(line_points[:, 0] - x_line) < 0.02]
        assert on_line.shape[0] >= 8  # fine edge voxel keeps the silhouette dense
        # and their observations are all edge pixels
        obs_ids = result["observations"]["fused_point_id"]
        is_edge_obs = result["observations"]["source_is_edge"]
        edge_fused_ids = np.flatnonzero(fused["is_edge"])
        sel = np.isin(obs_ids, edge_fused_ids)
        assert is_edge_obs[sel].mean() > 0.9

    def test_small_scenes_are_never_padded_or_overmerged(self):
        # 32 candidates total -> all must survive (degenerate budget branch)
        cfg = FusionConfig(target_points=1_000_000, min_points=10, max_points=2_000_000)
        view = _plane_view(0.0, (0, 0, 1), (0, 0, 5), image_id=1, size=4)
        result = fuse_views([view], cfg)
        assert result["fused"]["xyz"].shape[0] <= 16
        assert result["stats"]["below_target"] is True
        # every candidate is still represented in the provenance
        assert result["observations"]["fused_point_id"].size == 16

    def test_confidence_weighting_pulls_position(self):
        # two views observe the same pixel at different depths; the fused
        # point must land closer to the higher-confidence observation
        view_a = _plane_view(0.0, (0, 0, 1), (0, 0, 5), image_id=1, size=8, conf=9.0)
        view_b = _plane_view(0.10, (0, 0, 1), (0, 0, 5), image_id=2, size=8, conf=1.0)
        cfg = FusionConfig(target_points=64, min_points=2, max_points=10_000)
        result = fuse_views([view_a, view_b], cfg)
        zs = result["fused"]["xyz"][:, 2]
        # points merge (offset < voxel) and the mean is pulled towards z=0
        assert zs.mean() < 0.05 and zs.max() < 0.10


# --------------------------------------------------------------------------
# observation mapping
# --------------------------------------------------------------------------

class TestObservationMapping:
    def _table(self):
        return build_observation_table(
            fused_point_id=np.array([0, 0, 1, 1, 1]),
            source_image_id=np.array([1, 2, 1, 2, 1]),
            view_index=np.array([0, 1, 0, 1, 0]),
            source_pixel=np.array([10, 3, 7, 11, 12]),
            source_confidence=np.array([2.0, 3.0, 1.5, 2.5, 4.0]),
            source_normal=np.zeros((5, 3), dtype=np.float32),
            source_edge=np.array([False, True, False, False, True]),
            source_weight=np.array([2.0, 6.0, 1.5, 2.5, 8.0]),
            num_fused_points=2,
        )

    def test_roundtrip_npz_and_lookup(self):
        table = self._table()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "obs.npz")
            save_observations_npz(path, table, meta={"width": np.array([32])})
            loaded, meta = load_observations_npz(path)
            assert loaded["fused_point_id"].tolist() == table["fused_point_id"].tolist()
            assert meta["width"].tolist() == [32]
        subset = loaded if False else table
        assert subset["source_weight"].size == 5

    def test_out_of_range_point_id_raises(self):
        with pytest.raises(ValueError):
            build_observation_table(
                fused_point_id=np.array([0, 5]),
                source_image_id=np.array([1, 2]),
                view_index=np.array([0, 1]),
                source_pixel=np.array([0, 1]),
                source_confidence=np.array([1.0, 1.0]),
                source_normal=np.zeros((2, 3), dtype=np.float32),
                source_edge=np.array([False, False]),
                source_weight=np.array([1.0, 1.0]),
                num_fused_points=2,
            )

    def test_pixel_index_to_xy(self):
        x, y = pixel_index_to_xy(np.array([0, 5, 6, 12]), width=6)
        assert x.tolist() == [0, 5, 0, 0]
        assert y.tolist() == [0, 0, 1, 2]

    def test_csv_export_bounded(self):
        table = self._table()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "obs.csv")
            rows = export_observations_csv(path, table, max_rows=3)
            assert rows == 3
            with open(path, encoding="utf-8") as handle:
                lines = handle.read().strip().splitlines()
            assert len(lines) == 4  # header + 3 rows
            assert lines[0].startswith("fused_point_id")
