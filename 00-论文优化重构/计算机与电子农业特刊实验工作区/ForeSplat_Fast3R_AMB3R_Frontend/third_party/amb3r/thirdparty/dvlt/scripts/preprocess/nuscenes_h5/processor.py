# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NuScenes dataset preprocessor — HDF5 depth-map generation.

For each scene, generates a ``scene.h5`` containing:
- images/{idx}: JPEG-encoded image bytes for each CAM_FRONT keyframe
- depths: (num_frames, H, W) float32 sparse depth maps from LiDAR projection
- extrinsics_c2w: (num_frames, 4, 4) float64 camera-to-world matrices
- intrinsics: (num_frames, 3, 3) float64 camera intrinsic matrices

Processes annotated keyframes only (2 Hz).

Output structure per scene:
    {save_dir}/{scene_name}/
        scene.h5
"""

import gc
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
from nuscenes.nuscenes import LidarPointCloud, NuScenes
from pyquaternion import Quaternion
from tqdm import tqdm


# Lidarseg class indices to filter out when generating depth maps
LIDARSEG_SKIP_INDICES = {0, 31}  # 0 = noise, 31 = vehicle.ego

# NuScenes provides 6 surround cameras
CAMERA_LIST = [
    "CAM_FRONT",  # cam_id 0
    "CAM_FRONT_LEFT",  # cam_id 1
    "CAM_FRONT_RIGHT",  # cam_id 2
    "CAM_BACK_LEFT",  # cam_id 3
    "CAM_BACK_RIGHT",  # cam_id 4
    "CAM_BACK",  # cam_id 5
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_transform(rotation: list, translation: list) -> np.ndarray:
    """Build a 4x4 rigid-body transform from a NuScenes quaternion and translation."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = np.array(translation)
    return T


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class NuScenesProcessor:
    """Process NuScenes dataset into per-scene HDF5 files.

    Args:
        load_dir: Source directory of raw NuScenes data.
        save_dir: Target directory for processed data.
        split: Dataset split (e.g. 'v1.0-mini').
        process_id_list: Scene indices to process.
        workers: Number of parallel workers.
        lidarseg_root: Path to directory containing lidarseg.json and category.json.
            Optional: when provided, noise and ego-vehicle LiDAR points are filtered
            from depth maps. Without it, depth maps use all LiDAR points.
    """

    def __init__(
        self,
        load_dir: str,
        save_dir: str,
        split: str = "v1.0-mini",
        process_id_list: Optional[List[int]] = None,
        workers: int = 4,
        lidarseg_root: Optional[str] = None,
    ):
        self.process_id_list = process_id_list
        self.lidarseg_root = lidarseg_root

        # Build lidarseg token-to-filename index if available
        self._lidarseg_index: Optional[Dict[str, str]] = None
        if self.lidarseg_root is not None:
            self._lidarseg_index = self._load_lidarseg_index()
            print("Lidarseg filtering enabled (removing noise + ego vehicle points)")
        else:
            print("No --lidarseg_root provided; depth maps will use all LiDAR points")

        self.cam_list = CAMERA_LIST
        self.lidar_list = ["LIDAR_TOP"]

        self.load_dir = load_dir
        self.save_dir = os.path.join(save_dir, split.split("-")[-1])
        self.workers = int(workers)

        print(f"Output directory: {self.save_dir}")

        self.nusc = NuScenes(version=split, dataroot=load_dir, verbose=True)
        self._create_folders()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self):
        """Run conversion for all scenes.

        Scene index is stable for a given NuScenes version: nusc.scene order is fixed
        from the dataset, so the same index always refers to the same scene (same name).
        """
        print("Start converting ...")
        id_list = self.process_id_list if self.process_id_list is not None else list(range(len(self.nusc.scene)))

        todo_list = [i for i in id_list if not self._scene_already_processed(i)]
        skipped_list = [i for i in id_list if self._scene_already_processed(i)]
        skipped = len(skipped_list)
        if skipped > 0:
            names = [self._scene_dir_name(i) for i in skipped_list]
            if skipped <= 10:
                print(f"Skipping {skipped} already-processed scene(s): {', '.join(names)}")
            else:
                print(f"Skipping {skipped} already-processed scene(s): {', '.join(names[:5])}, ... ({skipped} total)")
        if not todo_list:
            print("Nothing to do.")
            return
        id_list = todo_list

        if self.workers <= 1:
            for scene_idx in tqdm(id_list, desc="Scenes"):
                self._convert_one(scene_idx)
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                results = list(
                    tqdm(
                        executor.map(self._convert_one_safe, id_list),
                        total=len(id_list),
                        desc="Scenes",
                    )
                )
            failures = [r for r in results if r is not None]
            if failures:
                for scene_idx, scene_name, err in failures:
                    print(f"FAILED scene {scene_idx} ({scene_name}): {err}")
                print(f"{len(failures)} scene(s) failed.")

        print("\nFinished.")

    # ------------------------------------------------------------------
    # Per-scene conversion
    # ------------------------------------------------------------------

    def _convert_one(self, scene_idx: int):
        """Process a single scene at the annotated keyframe rate (2 Hz)."""
        scene_data = self.nusc.get("scene", self.nusc.scene[scene_idx]["token"])
        scene_name = self.nusc.scene[scene_idx]["name"]
        self._save_depth_maps_h5(scene_data, scene_idx)
        print(f"Scene {scene_name} done.")

    def _convert_one_safe(self, scene_idx: int) -> Optional[Tuple[int, str, str]]:
        """Run _convert_one; return None on success, (scene_idx, scene_name, error) on failure."""
        scene_name = self.nusc.scene[scene_idx]["name"]
        try:
            self._convert_one(scene_idx)
            return None
        except Exception as e:
            return (scene_idx, scene_name, str(e))

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def _scene_dir_name(self, scene_idx: int) -> str:
        """Directory name for this scene (e.g. 'scene-0055')."""
        return self.nusc.scene[scene_idx]["name"]

    def _iter_keyframes(self, scene_data: dict) -> Generator[Tuple[int, dict], None, None]:
        """Yield ``(frame_idx, sample_record)`` for each annotated keyframe in a scene."""
        first_tok = scene_data["first_sample_token"]
        last_tok = scene_data["last_sample_token"]
        rec = self.nusc.get("sample", first_tok)
        frame_idx = 0
        while True:
            yield frame_idx, rec
            if rec["next"] == "" or rec["token"] == last_tok:
                break
            frame_idx += 1
            rec = self.nusc.get("sample", rec["next"])

    # ------------------------------------------------------------------
    # Skip-detection
    # ------------------------------------------------------------------

    def _scene_already_processed(self, scene_idx: int) -> bool:
        """Return True if this scene's scene.h5 already exists."""
        base = f"{self.save_dir}/{self._scene_dir_name(scene_idx)}"
        return os.path.isfile(f"{base}/scene.h5")

    # ------------------------------------------------------------------
    # Depth maps (HDF5 export)
    # ------------------------------------------------------------------

    def _load_lidarseg_index(self) -> Dict[str, str]:
        """Load lidarseg.json and build a sample_data_token -> filename mapping."""
        lidarseg_json_path = os.path.join(self.lidarseg_root, "lidarseg.json")
        with open(lidarseg_json_path, "r") as f:
            entries = json.load(f)
        return {entry["sample_data_token"]: entry["filename"] for entry in entries}

    def _save_depth_maps_h5(self, scene_data: dict, scene_idx: int):
        """Generate sparse depth maps for CAM_FRONT and write a per-scene HDF5 file.

        For each keyframe:
        1. Load LiDAR points (sensor frame) and optionally lidarseg labels
        2. If lidarseg available, filter out noise (idx 0) and ego vehicle (idx 31)
        3. Transform to camera frame via lidar_to_world and cam_to_world
        4. Project to image plane via intrinsics
        5. Z-buffer (minimum depth per pixel)

        Writes: ``{save_dir}/{scene_name}/scene.h5``
        """
        cam_name = self.cam_list[0]  # CAM_FRONT

        all_depths: List[np.ndarray] = []
        all_c2w: List[np.ndarray] = []
        all_K: List[np.ndarray] = []
        all_img_bytes: List[bytes] = []

        for _, rec in self._iter_keyframes(scene_data):
            # --- LiDAR data ---
            lidar_token = rec["data"][self.lidar_list[0]]
            lidar_data = self.nusc.get("sample_data", lidar_token)
            lidar_path = os.path.join(self.nusc.dataroot, lidar_data["filename"])
            pc = LidarPointCloud.from_file(lidar_path)
            pts = pc.points[:3, :].T  # (N, 3) in lidar sensor frame

            # Lidarseg filtering (optional): drop noise + ego-vehicle points
            if self._lidarseg_index is not None and lidar_token in self._lidarseg_index:
                seg_path = os.path.join(self.nusc.dataroot, self._lidarseg_index[lidar_token])
                labels = np.fromfile(seg_path, dtype=np.uint8)
                keep = np.array([lb not in LIDARSEG_SKIP_INDICES for lb in labels])
                pts = pts[keep]

            # Lidar-to-world
            lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            lidar_ego = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
            l2e = _build_transform(lidar_calib["rotation"], lidar_calib["translation"])
            e2w = _build_transform(lidar_ego["rotation"], lidar_ego["translation"])
            lidar_to_world = e2w @ l2e

            # Camera data (CAM_FRONT)
            cam_data = self.nusc.get("sample_data", rec["data"][cam_name])
            cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            cam_ego = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
            c2e = _build_transform(cam_calib["rotation"], cam_calib["translation"])
            e2w_cam = _build_transform(cam_ego["rotation"], cam_ego["translation"])
            cam_to_world = e2w_cam @ c2e
            world_to_cam = np.linalg.inv(cam_to_world)

            K = np.array(cam_calib["camera_intrinsic"])  # (3, 3)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            img_h, img_w = cam_data["height"], cam_data["width"]

            # Transform lidar points to camera frame
            lidar_to_cam = world_to_cam @ lidar_to_world
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])  # (N, 4)
            pts_cam = (lidar_to_cam @ pts_h.T).T[:, :3]  # (N, 3)

            # Filter points behind camera
            z = pts_cam[:, 2]
            pts_cam = pts_cam[z > 0.1]
            z = pts_cam[:, 2]

            # Project to pixel coordinates
            u = (fx * pts_cam[:, 0] / z + cx).astype(np.int32)
            v = (fy * pts_cam[:, 1] / z + cy).astype(np.int32)

            # Filter points outside image
            in_image = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
            u, v, z = u[in_image], v[in_image], z[in_image]

            # Z-buffer: keep minimum depth per pixel
            depth_map = np.full((img_h, img_w), np.inf, dtype=np.float32)
            for ui, vi, zi in zip(u, v, z):
                depth_map[vi, ui] = min(depth_map[vi, ui], zi)
            depth_map[depth_map == np.inf] = 0.0  # no-data -> 0

            all_depths.append(depth_map)
            all_c2w.append(cam_to_world)
            all_K.append(K)

            img_src = os.path.join(self.nusc.dataroot, cam_data["filename"])
            with open(img_src, "rb") as f:
                all_img_bytes.append(f.read())

        # --- Write HDF5 (stream frame-by-frame to limit peak memory) ---
        sid = self._scene_dir_name(scene_idx)
        h5_path = f"{self.save_dir}/{sid}/scene.h5"
        n_frames = len(all_depths)
        with h5py.File(h5_path, "w") as hf:
            h_depths = hf.create_dataset(
                "depths",
                shape=(n_frames,) + all_depths[0].shape,
                dtype=np.float32,
            )
            for i, d in enumerate(all_depths):
                h_depths[i] = d

            hf.create_dataset("extrinsics_c2w", data=np.stack(all_c2w, axis=0))
            hf.create_dataset("intrinsics", data=np.stack(all_K, axis=0))

            img_grp = hf.create_group("images")
            for idx, img_b in enumerate(all_img_bytes):
                img_grp.create_dataset(
                    str(idx),
                    data=np.frombuffer(img_b, dtype=np.uint8),
                )

        # Explicitly free large lists so worker memory is released before next scene
        del all_depths, all_c2w, all_K, all_img_bytes
        gc.collect()

        print(f"Saved depth maps HDF5: {h5_path} ({n_frames} frames)")

    # ------------------------------------------------------------------
    # Folder creation
    # ------------------------------------------------------------------

    def _create_folders(self):
        """Pre-create the output directory tree for every scene to be processed."""
        id_list = self.process_id_list if self.process_id_list is not None else list(range(len(self.nusc.scene)))
        for i in id_list:
            sid = self._scene_dir_name(i)
            base = f"{self.save_dir}/{sid}"
            os.makedirs(base, exist_ok=True)
