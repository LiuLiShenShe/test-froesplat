"""Interface-level tests for the ForeSplat VGGT-Omega frontend adapter.

Run with::

    python -m pytest tests/ -v          # or: python -m unittest discover tests -v

These tests do not need the VGGT-Omega checkpoint; they validate the
preprocessing geometry (RGB/mask consistency, pixel transforms, intrinsic
mapping) and the COLMAP export on synthetic predictions.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vggt_omega.utils.load_fn import (  # noqa: E402
    apply_pixel_transform,
    load_and_preprocess_images,
    load_and_preprocess_images_with_masks,
    model_to_original_coordinates,
    transform_intrinsics_model_to_original,
)
from visual_util import save_colmap  # noqa: E402

_INVALID_POINT3D_ID = 2 ** 64 - 1  # pycolmap<4 lacks INVALID_POINT3D_ID


def make_synthetic_dataset(tmpdir, sizes, seed=0):
    rng = np.random.default_rng(seed)
    image_paths, mask_paths = [], []
    for i, (w, h) in enumerate(sizes):
        arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        image_path = os.path.join(tmpdir, f"{i:06d}.png")
        Image.fromarray(arr).save(image_path)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
        mask_path = os.path.join(tmpdir, f"m{i:06d}.png")
        Image.fromarray(mask).save(mask_path)
        image_paths.append(image_path)
        mask_paths.append(mask_path)
    return image_paths, mask_paths


class TestPreprocessGeometry(unittest.TestCase):
    """The mask must follow the exact RGB geometry; A must match PIL resize."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.sizes = [(1200, 400), (800, 600), (600, 800), (300, 700)]
        cls.image_paths, cls.mask_paths = make_synthetic_dataset(cls.tmpdir, cls.sizes)
        cls.resolution = 512
        cls.images, cls.masks, cls.transforms = load_and_preprocess_images_with_masks(
            cls.image_paths, cls.mask_paths, image_resolution=cls.resolution
        )

    def test_rgb_path_identical_to_community_loader(self):
        community = load_and_preprocess_images(self.image_paths, image_resolution=self.resolution)
        torch.testing.assert_close(self.images, community, rtol=0, atol=0)

    def test_shapes_and_mask_values(self):
        n, _, h, w = self.images.shape
        self.assertEqual(self.masks.shape, (n, h, w))
        self.assertEqual(self.masks.dtype, torch.bool)
        self.assertTrue(set(np.unique(self.masks.numpy()).tolist()) <= {False, True})
        for record in self.transforms:
            self.assertEqual(record["model_width"], w)
            self.assertEqual(record["model_height"], h)
            self.assertEqual(record["patch_size"], 16)
            self.assertEqual(record["resized_width"] + record["pad_left"] + record["pad_right"], w)
            self.assertEqual(record["resized_height"] + record["pad_top"] + record["pad_bottom"], h)

    def test_no_foreground_in_padding(self):
        for i, record in enumerate(self.transforms):
            m = self.masks[i].numpy()
            if record["pad_top"]:
                self.assertEqual(m[: record["pad_top"]].sum(), 0)
            if record["pad_bottom"]:
                self.assertEqual(m[-record["pad_bottom"] :].sum(), 0)
            if record["pad_left"]:
                self.assertEqual(m[:, : record["pad_left"]].sum(), 0)
            if record["pad_right"]:
                self.assertEqual(m[:, -record["pad_right"] :].sum(), 0)

    def test_mask_foreground_ratio_matches_resize(self):
        # Nearest resampling preserves the mask fraction *of the cropped region*
        # (padding is background and excluded here).
        for i in range(len(self.sizes)):
            record = self.transforms[i]
            w0, h0 = record["original_width"], record["original_height"]
            cl, ct, cw, ch = (
                record["crop_left"],
                record["crop_top"],
                record["crop_width"],
                record["crop_height"],
            )
            x0, x1, y0, y1 = w0 // 4, 3 * w0 // 4, h0 // 4, 3 * h0 // 4
            inter_w = max(0, min(x1, cl + cw) - max(x0, cl))
            inter_h = max(0, min(y1, ct + ch) - max(y0, ct))
            expected = (inter_w * inter_h) / (cw * ch)
            m = self.masks[i].numpy()
            top, left = record["pad_top"], record["pad_left"]
            valid = m[top : top + record["resized_height"], left : left + record["resized_width"]]
            self.assertAlmostEqual(float(valid.mean()), expected, delta=0.02)

    def test_pixel_transform_matches_pil_resize(self):
        # A single bright pixel (landmark) in the original must land at A @ p after
        # the real crop -> bicubic resize -> pad chain (centroid check).
        tmpdir = tempfile.mkdtemp()
        w0, h0 = 811, 617
        arr = np.zeros((h0, w0, 3), dtype=np.uint8)
        x0, y0 = 521, 311
        arr[y0, x0] = 255
        image_path = os.path.join(tmpdir, "landmark.png")
        Image.fromarray(arr).save(image_path)
        mask_path = os.path.join(tmpdir, "landmark_mask.png")
        Image.fromarray(np.zeros((h0, w0), dtype=np.uint8)).save(mask_path)

        images, masks, records = load_and_preprocess_images_with_masks(
            [image_path], [mask_path], image_resolution=512
        )
        record = records[0]
        mapped = apply_pixel_transform(record["pixel_transform"], [[x0, y0]])[0]

        blob = images[0].numpy().sum(axis=0)
        ys, xs = np.nonzero(blob > 0)
        weights = blob[ys, xs]
        centroid = np.array([np.average(xs, weights=weights), np.average(ys, weights=weights)])
        self.assertTrue(np.allclose(mapped, centroid, atol=0.75), f"A@p={mapped} vs centroid={centroid}")

        # Inverse transform of the centroid must return to the original landmark.
        back = model_to_original_coordinates(record["pixel_transform"], centroid[None])[0]
        self.assertTrue(np.allclose(back, [x0, y0], atol=0.75), f"A^-1@centroid={back} vs ({x0},{y0})")

    def test_nearest_mask_mapping_roundtrip(self):
        # Every foreground pixel in the model mask must map back into the foreground
        # rectangle of the original mask.
        for i, record in enumerate(self.transforms):
            w0, h0 = record["original_width"], record["original_height"]
            rect = (w0 // 4, h0 // 4, 3 * w0 // 4, 3 * h0 // 4)
            ys, xs = np.nonzero(self.masks[i].numpy())
            if len(xs) == 0:
                continue
            pts = model_to_original_coordinates(record["pixel_transform"], np.stack([xs, ys], axis=1))
            inside = (
                (pts[:, 0] >= rect[0] - 1.0)
                & (pts[:, 0] <= rect[2] + 1.0)
                & (pts[:, 1] >= rect[1] - 1.0)
                & (pts[:, 1] <= rect[3] + 1.0)
            )
            self.assertGreaterEqual(inside.mean(), 0.98)

    def test_rgba_composited_over_black(self):
        tmpdir = tempfile.mkdtemp()
        arr = np.zeros((800, 640, 4), dtype=np.uint8)
        arr[..., :3] = 200
        arr[150:550, 120:520, 3] = 255
        rgba_path = os.path.join(tmpdir, "rgba.png")
        Image.fromarray(arr, mode="RGBA").save(rgba_path)
        mask_path = os.path.join(tmpdir, "rgba_mask.png")
        Image.fromarray((arr[..., 3]).astype(np.uint8)).save(mask_path)

        images, masks, records = load_and_preprocess_images_with_masks(
            [rgba_path], [mask_path], image_resolution=512, background="black"
        )
        record = records[0]
        rgb = images[0].numpy().transpose(1, 2, 0)
        top, left = record["pad_top"], record["pad_left"]
        valid_rgb = rgb[top : top + record["resized_height"], left : left + record["resized_width"]]
        valid_mask = masks[0].numpy()[top : top + record["resized_height"], left : left + record["resized_width"]]
        # Bicubic resampling bleeds a few pixels across the mask boundary, so compare
        # only pixels at least 4 px away from the mask edge.
        from scipy.ndimage import binary_dilation, binary_erosion

        outside_core = ~binary_dilation(valid_mask, iterations=4)
        inside_core = binary_erosion(valid_mask, iterations=4)
        # outside the alpha region everything must be pure black (RAP-FSAM3 foreground form)
        self.assertLessEqual(float(valid_rgb[outside_core].max()), 1.0 / 255.0)
        # inside, the colour must be preserved (200/255)
        self.assertGreater(int(inside_core.sum()), 0)
        self.assertTrue(
            np.allclose(valid_rgb[inside_core], 200.0 / 255.0, atol=4.0 / 255.0),
            f"got {valid_rgb[inside_core].min()}..{valid_rgb[inside_core].max()}",
        )


class _SyntheticPredictions:
    """Deterministic camera/depth predictions consistent with the loader records."""

    @staticmethod
    def build(transforms, num_frames, height, width, seed=0):
        rng = np.random.default_rng(seed)
        intrinsics = np.zeros((num_frames, 3, 3))
        extrinsics = np.zeros((num_frames, 3, 4))
        for i in range(num_frames):
            fx = 0.8 * width
            fy = 0.8 * height
            intrinsics[i] = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
            # Small distinct camera poses: rotate around Y, translate on a circle.
            theta = 2 * np.pi * i / max(num_frames, 1)
            R = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
            t = -R @ np.array([2.0 * np.cos(theta), 0.1 * i, 2.0 * np.sin(theta)])
            extrinsics[i, :3, :3] = R
            extrinsics[i, :3, 3] = t

        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        depth = 2.0 + 0.3 * xx / width + 0.1 * yy / height  # (H, W)
        depth = np.stack([depth] * num_frames)  # (N, H, W)
        conf = 1.0 + rng.random((num_frames, height, width))

        # world_points_from_depth must be generated exactly like inference does.
        from foresplat_adapter.inference import unproject_depth_map_to_point_map

        world_points = unproject_depth_map_to_point_map(depth[..., None], extrinsics, intrinsics)

        images = rng.random((num_frames, 3, height, width))
        predictions = {
            "extrinsic": extrinsics,
            "intrinsic": intrinsics,
            "depth": depth[..., None],
            "depth_conf": conf,
            "images": images,
            "world_points_from_depth": world_points,
        }
        return predictions


class TestSaveColmap(unittest.TestCase):
    """ForeSplat COLMAP export: masks, transforms, intrinsics and observations."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.sizes = [(900, 500), (600, 800)]
        cls.image_paths, cls.mask_paths = make_synthetic_dataset(cls.tmpdir, cls.sizes, seed=1)
        cls.images, cls.masks, cls.transforms = load_and_preprocess_images_with_masks(
            cls.image_paths, cls.mask_paths, image_resolution=512
        )
        num_frames, _, height, width = cls.images.shape
        cls.predictions = _SyntheticPredictions.build(cls.transforms, num_frames, height, width)
        cls.output_dir = os.path.join(cls.tmpdir, "GGGS_Dataset")
        os.makedirs(cls.output_dir, exist_ok=True)

        saved_image_names = []
        for i, path in enumerate(cls.image_paths):
            dst = os.path.join(cls.output_dir, "images", os.path.basename(path))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            Image.open(path).save(dst)
            saved_image_names.append(dst)

        cls.stats = save_colmap(
            cls.predictions,
            cls.output_dir,
            image_names=saved_image_names,
            conf_thres=20.0,
            max_points=100000,
            foreground_masks=cls.masks.numpy(),
            preprocess_transforms=cls.transforms,
            filter_depth_edges=False,
            colmap_subdir="sparse/0",
            apply_alignment=False,  # pure geometry check first
            seed=0,
            return_stats=True,
        )

    def test_point_count_and_per_frame_counts(self):
        self.assertEqual(self.stats["num_frames"], len(self.sizes))
        total = sum(self.stats["per_frame_point2d"])
        self.assertEqual(self.stats["num_points3d"], total)
        for count in self.stats["per_frame_point2d"]:
            self.assertGreater(count, 0)

    def test_conf_threshold_computed_on_foreground_only(self):
        # All confidences are >= 1.0; the 20th percentile threshold must be within
        # the foreground confidence range (not the full-frame range).
        fg_conf = self.predictions["depth_conf"][self.masks.numpy()]
        self.assertGreaterEqual(self.stats["conf_threshold_value"], fg_conf.min())
        self.assertLessEqual(self.stats["conf_threshold_value"], np.percentile(fg_conf, 20.0001))

    def test_colmap_files_written(self):
        sparse = os.path.join(self.output_dir, "sparse", "0")
        for name in ("cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"):
            self.assertTrue(os.path.isfile(os.path.join(sparse, name)), name)

    def test_tracks_have_single_observation(self):
        import pycolmap

        rec = pycolmap.Reconstruction(os.path.join(self.output_dir, "sparse", "0"))
        self.assertEqual(rec.num_points3D(), self.stats["num_points3d"])
        for pid, point in rec.points3D.items():
            self.assertEqual(point.track.length(), 1, f"point {pid} must have exactly one observation")

    def test_intrinsics_and_observations_consistent(self):
        """K_original = A^-1 K_model must make every observation reproject its own point."""
        import pycolmap

        rec = pycolmap.Reconstruction(os.path.join(self.output_dir, "sparse", "0"))
        max_err = 0.0
        for image_id, image in rec.images.items():
            camera = rec.cameras[image.camera_id]
            fx, fy, cx, cy = np.asarray(camera.params, dtype=np.float64)
            R = image.cam_from_world.rotation.matrix()
            t = np.asarray(image.cam_from_world.translation, dtype=np.float64)
            for point2D in image.points2D:
                pid = point2D.point3D_id
                if pid >= _INVALID_POINT3D_ID - 1:
                    continue
                xyz = np.asarray(rec.points3D[pid].xyz, dtype=np.float64)
                cam = R @ xyz + t
                self.assertGreater(cam[2], 0)
                proj = np.array([fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy])
                err = np.linalg.norm(proj - np.asarray(point2D.xy, dtype=np.float64))
                max_err = max(max_err, float(err))
        self.assertLess(max_err, 1e-5, f"max reprojection error {max_err}")

    def test_observations_inside_original_image(self):
        import pycolmap

        rec = pycolmap.Reconstruction(os.path.join(self.output_dir, "sparse", "0"))
        for image_id, image in rec.images.items():
            camera = rec.cameras[image.camera_id]
            for point2D in image.points2D:
                self.assertGreaterEqual(point2D.xy[0], 0.0)
                self.assertGreaterEqual(point2D.xy[1], 0.0)
                self.assertLess(point2D.xy[0], camera.width)
                self.assertLess(point2D.xy[1], camera.height)

    def test_scene_alignment_preserves_consistency(self):
        aligned_dir = os.path.join(self.tmpdir, "aligned")
        os.makedirs(aligned_dir, exist_ok=True)
        stats = save_colmap(
            self.predictions,
            aligned_dir,
            image_names=self.image_paths,
            conf_thres=20.0,
            max_points=100000,
            foreground_masks=self.masks.numpy(),
            preprocess_transforms=self.transforms,
            filter_depth_edges=False,
            colmap_subdir="sparse/0",
            apply_alignment=True,
            seed=0,
            return_stats=True,
        )
        import pycolmap

        rec = pycolmap.Reconstruction(os.path.join(aligned_dir, "sparse", "0"))
        max_err = 0.0
        for image_id, image in rec.images.items():
            camera = rec.cameras[image.camera_id]
            fx, fy, cx, cy = np.asarray(camera.params, dtype=np.float64)
            R = image.cam_from_world.rotation.matrix()
            t = np.asarray(image.cam_from_world.translation, dtype=np.float64)
            for point2D in image.points2D:
                pid = point2D.point3D_id
                if pid >= _INVALID_POINT3D_ID - 1:
                    continue
                xyz = np.asarray(rec.points3D[pid].xyz, dtype=np.float64)
                cam = R @ xyz + t
                proj = np.array([fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy])
                err = np.linalg.norm(proj - np.asarray(point2D.xy, dtype=np.float64))
                max_err = max(max_err, float(err))
        self.assertLess(max_err, 1e-5)
        # The alignment must be a proper rotation (det=+1), no double flip.
        T = np.asarray(stats["scene_alignment_matrix"], dtype=np.float64)
        self.assertAlmostEqual(np.linalg.det(T[:3, :3]), 1.0, places=6)

    def test_intrinsic_scaling_matches_transform(self):
        import pycolmap

        rec = pycolmap.Reconstruction(os.path.join(self.output_dir, "sparse", "0"))
        for fidx, image_id in enumerate(sorted(rec.images)):
            image = rec.images[image_id]
            camera = rec.cameras[image.camera_id]
            params = np.asarray(camera.params, dtype=np.float64)
            K_orig_expected = transform_intrinsics_model_to_original(
                self.predictions["intrinsic"][fidx], self.transforms[fidx]["pixel_transform"]
            )
            expected = np.array(
                [K_orig_expected[0, 0], K_orig_expected[1, 1], K_orig_expected[0, 2], K_orig_expected[1, 2]]
            )
            self.assertTrue(np.allclose(params, expected, atol=1e-6), f"{params} vs {expected}")
            # camera size must be the original resolution
            self.assertEqual((camera.width, camera.height), (self.transforms[fidx]["original_width"], self.transforms[fidx]["original_height"]))


if __name__ == "__main__":
    unittest.main()


class TestDatasetPairing(unittest.TestCase):
    """RAP-FSAM3 run directories use different name prefixes (crop_ / mask_)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.tmpdir, "images")
        self.masks_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(self.images_dir)
        os.makedirs(self.masks_dir)
        rng = np.random.default_rng(3)
        for i in range(4):
            Image.fromarray(rng.integers(0, 255, (64, 48, 3), dtype=np.uint8)).save(
                os.path.join(self.images_dir, f"crop_{i:04d}.png")
            )
            Image.fromarray(np.full((64, 48), 255, dtype=np.uint8)).save(
                os.path.join(self.masks_dir, f"mask_{i:04d}.png")
            )

    def _collect(self, **kwargs):
        from foresplat_adapter.dataset import collect_frames

        return collect_frames(self.images_dir, self.masks_dir, **kwargs)

    def test_pairing_via_prefix_strip(self):
        frames = self._collect(image_prefix="crop_", mask_prefix="mask_")
        self.assertEqual(len(frames), 4)
        for i, frame in enumerate(frames):
            self.assertEqual(frame.image_name, f"crop_{i:04d}.png")
            self.assertTrue(os.path.basename(frame.mask_path) == f"mask_{i:04d}.png")

    def test_pairing_via_tail_digit_fallback(self):
        # Without prefixes the exact stem differs; the trailing digit run must pair them.
        frames = self._collect()
        self.assertEqual(len(frames), 4)
        for frame in frames:
            self.assertTrue(os.path.isfile(frame.mask_path))

    def test_missing_mask_raises(self):
        os.remove(os.path.join(self.masks_dir, "mask_0002.png"))
        with self.assertRaises(ValueError):
            self._collect()

    def test_order_and_subset_preserved(self):
        frames = self._collect(frame_stride=2, max_frames=2)
        self.assertEqual([f.image_name for f in frames], ["crop_0000.png", "crop_0002.png"])
        self.assertEqual([f.index for f in frames], [0, 1])
