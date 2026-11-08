"""Regression tests for the edge-thinning provenance path of fuse_views."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresplat_adapter.fusion import FusionConfig, fuse_views  # noqa: E402
from tests.test_fusion import _plane_view  # noqa: E402


def _two_views_with_edges(size: int = 32):
    view_a = _plane_view(0.0, (0, 0, 1), (0, 0, 5), image_id=1, size=size, seed=1)
    view_b = _plane_view(0.0, (0, 0, 1), (1, 0, 5), image_id=2, size=size, seed=2)
    # mark a thick band of pixels as depth discontinuities
    view_a["edge"][:, 8:24] = True
    view_b["edge"][:, 8:24] = True
    return view_a, view_b


class TestEdgeThinningProvenance:
    def test_thinned_observations_are_never_misattributed(self):
        views = _two_views_with_edges()
        # Keep the *core* pass small so the hard cap (max_points) has room to
        # bite only on the edge pass, which is the part that gets thinned.
        cfg = FusionConfig(
            target_points=2000, min_points=2, max_points=240,
            core_fraction=0.1, edge_voxel_divisor=1.0, edge_weight_boost=2.0,
            seed=0,
        )
        result = fuse_views(views, cfg)
        fused = result["fused"]
        obs = result["observations"]
        stats = result["stats"]

        assert stats["num_fused_points"] <= cfg.max_points
        # thinning must have triggered (edge pass exists and was capped)
        assert stats["edge_points"] > 0
        assert stats["thinned_edge_points"] > 0
        assert stats["observations_dropped"] > 0
        assert stats["observations_kept"] + stats["observations_dropped"] == stats["num_candidates"]

        # every surviving observation points at a real fused point
        assert obs["fused_point_id"].min() >= 0
        assert obs["fused_point_id"].max() < fused["xyz"].shape[0]

        # no observation of a *dropped* edge group survives
        counts = np.bincount(obs["fused_point_id"], minlength=fused["xyz"].shape[0])
        assert counts.min() >= 1
        # every fused point that has observations is consistent with its count
        assert counts.sum() == stats["observations_kept"]

        # edge representatives that survived are all reachable from provenance
        edge_ids = np.flatnonzero(fused["is_edge"])
        covered = np.unique(obs["fused_point_id"])
        assert np.isin(edge_ids, covered).all()

    def test_no_thinning_keeps_all_observations(self):
        views = _two_views_with_edges(size=16)
        cfg = FusionConfig(target_points=300, min_points=2, max_points=100_000, seed=0)
        result = fuse_views(views, cfg)
        stats = result["stats"]
        assert stats["thinned_edge_points"] == 0
        assert stats["observations_dropped"] == 0
        assert stats["observations_kept"] == stats["num_candidates"]
