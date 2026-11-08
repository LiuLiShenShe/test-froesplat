"""Integration tests: full-sequence fusion -> GGGS export with a fused points3D.ply."""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest

import numpy as np
import pycolmap
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from foresplat_adapter.export import export_gggs_dataset  # noqa: E402
from foresplat_adapter.fusion import FusionConfig, fuse_views  # noqa: E402
from foresplat_adapter.full_sequence import build_fusion_views  # noqa: E402
from foresplat_adapter.fusion.observation_mapping import (  # noqa: E402
    load_observations_npz,
    observations_for_point,
)
from tests.test_preprocessing_and_export import (  # noqa: E402
    _SyntheticPredictions,
    make_synthetic_dataset,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images_with_masks  # noqa: E402


class TestFusionExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.image_paths, cls.mask_paths = make_synthetic_dataset(cls.tmpdir, [(600, 800)] * 3, seed=3)
        cls.images, cls.masks, cls.transforms = load_and_preprocess_images_with_masks(
            cls.image_paths, cls.mask_paths, image_resolution=512
        )
        num_frames, _, height, width = cls.images.shape
        cls.predictions = _SyntheticPredictions.build(cls.transforms, num_frames, height, width)
        extrinsic = np.asarray(cls.predictions["extrinsic"], dtype=np.float64)

        views, cls.conf_threshold = build_fusion_views(
            cls.predictions, cls.masks.numpy(), extrinsic, conf_thres=20.0,
        )
        cls.fusion = fuse_views(views, FusionConfig(target_points=400, min_points=2, max_points=2000))

        cls.output_dir = os.path.join(cls.tmpdir, "GGGS_Dataset")
        os.makedirs(cls.output_dir, exist_ok=True)
        saved_image_names = []
        for path in cls.image_paths:
            dst = os.path.join(cls.output_dir, "images", os.path.basename(path))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            Image.open(path).save(dst)
            saved_image_names.append(dst)

        fused = cls.fusion["fused"]
        cls.summary = export_gggs_dataset(
            cls.predictions,
            cls.transforms,
            cls.masks,
            _FakeFrames(saved_image_names, cls.mask_paths),
            cls.output_dir,
            conf_thres=20.0,
            max_points=10_000_000,
            image_resolution=512,
            preprocess_mode="balanced",
            fused_points={
                "xyz": fused["xyz"].astype(np.float64),
                "colors": fused["colors"],
                "normals": fused["normals"].astype(np.float64),
                "is_edge": fused["is_edge"],
                "observation_count": fused["observation_count"],
                "confidence": fused["confidence"],
            },
            observations=cls.fusion["observations"],
        )

        with open(os.path.join(cls.output_dir, "sparse", "0", "points3D.ply"), "rb") as handle:
            data = handle.read()
        header_end = data.index(b"end_header\n") + len(b"end_header\n")
        cls.ply_header = data[:header_end].decode("ascii")
        cls.ply_bytes = data[header_end:]

    def test_points3d_ply_is_fused_cloud(self):
        n_fused = self.summary["num_fused_points_in_ply"]
        self.assertTrue(self.summary["points3d_ply_is_fused"])
        self.assertEqual(n_fused, self.fusion["fused"]["xyz"].shape[0])
        self.assertIn("element vertex %d" % n_fused, self.ply_header)
        for field in ("property float x", "property float nx", "property uchar red"):
            self.assertIn(field, self.ply_header)
        # fused cloud must be smaller than the raw per-view candidates
        self.assertLess(n_fused, self.summary["num_points3d"])

    def test_ply_payload_matches_fused_count(self):
        # 9 float/uchar fields per vertex: x,y,z,nx,ny,nz (4B each) + r,g,b (1B each)
        n = self.summary["num_fused_points_in_ply"]
        expected = n * (6 * 4 + 3)
        self.assertEqual(len(self.ply_bytes), expected)

    def test_raw_cloud_still_available(self):
        raw = os.path.join(self.output_dir, "debug", "raw_pointcloud.ply")
        self.assertTrue(os.path.isfile(raw))
        with open(raw, "rb") as handle:
            head = handle.read(1024).decode("ascii", "ignore")
        self.assertIn(f"element vertex {self.summary['num_points3d']}", head)

    def test_colmap_reconstruction_still_single_observation(self):
        rec = pycolmap.Reconstruction(os.path.join(self.output_dir, "sparse", "0"))
        self.assertEqual(rec.num_points3D(), self.summary["num_points3d"])
        for point in rec.points3D.values():
            self.assertEqual(point.track.length(), 1)

    def test_observations_npz_provenance(self):
        npz_path = os.path.join(self.output_dir, "debug", "observations.npz")
        observations, _ = load_observations_npz(npz_path)
        fused_ids = observations["fused_point_id"]
        self.assertEqual(fused_ids.max(), self.summary["num_fused_points_in_ply"] - 1)
        self.assertEqual(fused_ids.size, self.fusion["stats"]["num_candidates"])
        # every fused point is covered
        counts = np.bincount(fused_ids, minlength=self.summary["num_fused_points_in_ply"])
        self.assertEqual(counts.min(), 1)
        subset = observations_for_point(observations, 0)
        self.assertGreaterEqual(subset["fused_point_id"].size, 1)

    def test_fusion_metadata_written(self):
        import json

        path = os.path.join(self.output_dir, "metadata", "fusion_metadata.json")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as handle:
            meta = json.load(handle)
        self.assertTrue(meta["points3d_ply_is_fused"])
        self.assertEqual(meta["num_fused_points_in_ply"], self.summary["num_fused_points_in_ply"])
        self.assertEqual(meta["num_raw_per_view_points"], self.summary["num_points3d"])


class _FakeFrames:
    """Minimal stand-in for export's ``frames`` argument."""

    def __init__(self, image_paths, mask_paths):
        self._frames = [
            type(
                "F", (), {
                    "image_name": os.path.basename(p),
                    "image_path": p,
                    "mask_path": m,
                    "index": i,
                    "stem": os.path.splitext(os.path.basename(p))[0],
                },
            )
            for i, (p, m) in enumerate(zip(image_paths, mask_paths))
        ]

    def __iter__(self):
        return iter(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    def __len__(self):
        return len(self._frames)