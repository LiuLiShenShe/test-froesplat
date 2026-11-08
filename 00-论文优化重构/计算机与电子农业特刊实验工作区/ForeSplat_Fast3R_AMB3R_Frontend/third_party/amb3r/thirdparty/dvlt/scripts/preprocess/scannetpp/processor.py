# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScanNet++ DSLR preprocessing: render mesh depth and undistort to a pinhole model.

Per frame, depth is rasterized from the aligned GT mesh through the original
fisheye camera with PyTorch3D, then both the RGB and the depth are rectified to a
pinhole model with ``cv2.fisheye``. Depth uses nearest-neighbour resampling so the
rectified depth stays consistent with the fisheye-rendered values.

Output per scene, under ``{output_root}/{subset}/scannetpp/scannetpp_undistort/{scene}/``::

    undistorted_images/<name>.JPG
    undistorted_depth/<name>.png            # 16-bit, float16 bit-pattern (see common.io.read_depth)
    nerfstudio/transforms_undistorted.json  # PINHOLE intrinsics, original poses

``nvs_sem_train`` maps to subset ``train`` and ``nvs_sem_val`` to ``test`` to match
the dataset configs.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, fisheyecameras
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from scipy.spatial.transform import Rotation
from tqdm import tqdm


SPLIT_TO_SUBSET = {"nvs_sem_train": "train", "nvs_sem_val": "test"}


def read_colmap_poses(images_txt: Path) -> Dict[str, np.ndarray]:
    """Map image name -> 4x4 world-to-camera (OpenCV) matrix from a COLMAP ``images.txt``.

    Each posed image occupies one header row (quaternion + translation + name)
    followed by a keypoint row; only the header rows are needed here.
    """
    rows = [r for r in images_txt.read_text().splitlines() if r and not r.startswith("#")]

    poses: Dict[str, np.ndarray] = {}
    for header in rows[::2]:
        fields = header.split()
        qw, qx, qy, qz = (float(v) for v in fields[1:5])
        translation = [float(v) for v in fields[5:8]]
        extrinsic = np.eye(4)
        # COLMAP stores the quaternion scalar-first; scipy expects scalar-last.
        extrinsic[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        extrinsic[:3, 3] = translation
        poses[fields[9]] = extrinsic
    return poses


def rectified_intrinsics(K: np.ndarray, distortion: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics fitted to the rectified fisheye image, principal point at the centre."""
    fitted = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, distortion, (width, height), np.eye(3), balance=0.0
    )
    return np.array(
        [
            [fitted[0, 0], 0.0, width / 2.0],
            [0.0, fitted[1, 1], height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def make_fisheye_camera(pose_w2c: np.ndarray, K: np.ndarray, distortion: np.ndarray, height: int, width: int):
    """PyTorch3D ``FishEyeCameras`` for an OpenCV world-to-camera pose and OPENCV_FISHEYE coeffs."""
    rot_cv = torch.tensor(pose_w2c[:3, :3], dtype=torch.float32)
    trans_cv = torch.tensor(pose_w2c[:3, 3], dtype=torch.float32)
    img_size = torch.tensor([[height, width]], dtype=torch.float32)

    # PyTorch3D ships an OpenCV importer for pinhole cameras; borrow it for the NDC
    # focal length / principal point, which FishEyeCameras expects in the same units.
    pinhole = cameras_from_opencv_projection(
        R=rot_cv[None],
        tvec=trans_cv[None],
        camera_matrix=torch.tensor(K, dtype=torch.float32)[None],
        image_size=img_size,
    )

    # OpenCV -> PyTorch3D camera frame: row-vector rotation (transpose) with the X/Y axes flipped.
    rot = rot_cv.T.clone()
    rot[:, :2] *= -1.0
    trans = torch.tensor([-trans_cv[0], -trans_cv[1], trans_cv[2]], dtype=torch.float32)

    # FishEyeCameras uses a 6-term radial model; OPENCV_FISHEYE has 4 (k1..k4), pad the rest with 0.
    radial = torch.tensor([[*distortion, 0.0, 0.0]], dtype=torch.float32)

    return fisheyecameras.FishEyeCameras(
        focal_length=pinhole.focal_length,
        principal_point=pinhole.principal_point,
        radial_params=radial,
        use_radial=True,
        use_tangential=False,
        use_thin_prism=False,
        R=rot[None],
        T=trans[None],
        image_size=img_size,
        world_coordinates=True,
    )


def encode_depth_png(path: Path, depth: np.ndarray) -> None:
    """Write metric depth as a 16-bit PNG holding the float16 bit-pattern (read_depth inverse)."""
    half = np.ascontiguousarray(depth.astype(np.float16))
    # compress_level=1 is lossless (identical decoded values) but much faster than the default.
    Image.fromarray(half.view(np.uint16)).save(str(path), compress_level=1)


class ScanNetppProcessor:
    """Render depth and undistort DSLR captures for a set of ScanNet++ splits.

    Args:
        data_root: Raw release root containing ``data/`` and ``splits/``.
        output_root: Processed dataset root; this is the ``user.data_root`` the
            dataset configs point at. Outputs are written to
            ``{output_root}/{train,test}/scannetpp/...``.
        splits: Split names to process (subset of ``SPLIT_TO_SUBSET``).
        scene_ids: Optional explicit scene subset (default: all scenes in each split).
        device: Torch device for rasterization.
        overwrite: Re-render scenes even if their outputs already exist.
        bin_size: PyTorch3D rasterization bin size (default: None, the safe heuristic).
        max_faces_per_bin: PyTorch3D per-bin face budget (default: None, the safe
            heuristic). The default is correct but slow; a smaller budget (with a
            suitable ``bin_size``) renders much faster, but a budget that is too small
            silently drops faces and corrupts depth. Tune for your mesh size and GPU.
    """

    def __init__(
        self,
        data_root: str,
        output_root: str,
        splits: List[str],
        scene_ids: Optional[List[str]] = None,
        device: str = "cuda",
        overwrite: bool = False,
        bin_size: Optional[int] = None,
        max_faces_per_bin: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.splits = splits
        self.scene_ids = set(scene_ids) if scene_ids else None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.overwrite = overwrite
        self.bin_size = bin_size
        self.max_faces_per_bin = max_faces_per_bin

    def run(self) -> None:
        print(f"device: {self.device}  output_root: {self.output_root}")
        for split in self.splits:
            subset = SPLIT_TO_SUBSET[split]
            split_file = self.data_root / "splits" / f"{split}.txt"
            scene_list = [s.strip() for s in split_file.read_text().splitlines() if s.strip()]
            if self.scene_ids is not None:
                scene_list = [s for s in scene_list if s in self.scene_ids]

            out_base = self.output_root / subset / "scannetpp"
            undistort_root = out_base / "scannetpp_undistort"
            undistort_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(split_file, out_base / f"{split}.txt")

            print(f"[{split} -> {subset}] {len(scene_list)} scenes")
            for scene_id in tqdm(scene_list, desc=split):
                try:
                    self._process_scene(scene_id, undistort_root)
                except Exception as e:  # noqa: BLE001 - keep going, report at the end
                    print(f"FAILED {scene_id}: {e}")

        print(f"\nDone. Set `user.data_root` to: {self.output_root}")

    def _process_scene(self, scene_id: str, undistort_root: Path) -> None:
        scene_dir = self.data_root / "data" / scene_id / "dslr"
        if not scene_dir.exists():
            print(f"skip {scene_id}: no dslr/ directory")
            return

        out_dir = undistort_root / scene_id
        out_image_dir = out_dir / "undistorted_images"
        out_depth_dir = out_dir / "undistorted_depth"
        out_transforms = out_dir / "nerfstudio" / "transforms_undistorted.json"

        if not self.overwrite and out_transforms.exists() and any(out_image_dir.glob("*")):
            return

        transforms = json.loads((scene_dir / "nerfstudio" / "transforms.json").read_text())
        frames = transforms["frames"] + transforms.get("test_frames", [])

        K = np.array(
            [
                [float(transforms["fl_x"]), 0, float(transforms["cx"])],
                [0, float(transforms["fl_y"]), float(transforms["cy"])],
                [0, 0, 1],
            ]
        )
        distortion = np.array([float(transforms[k]) for k in ("k1", "k2", "k3", "k4")])
        width, height = int(transforms["w"]), int(transforms["h"])

        new_K = rectified_intrinsics(K, distortion, width, height)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

        poses = read_colmap_poses(scene_dir / "colmap" / "images.txt")
        mesh = self._load_mesh(scene_id)
        rasterizer = MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=(height, width),
                blur_radius=0.0,
                faces_per_pixel=1,
                cull_to_frustum=True,
                bin_size=self.bin_size,
                max_faces_per_bin=self.max_faces_per_bin,
            )
        )

        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_depth_dir.mkdir(parents=True, exist_ok=True)

        names = [
            f["file_path"]
            for f in frames
            if f["file_path"] in poses and (scene_dir / "resized_images" / f["file_path"]).exists()
        ]

        processed = 0
        for name in tqdm(names, desc=scene_id, unit="frame", leave=False):
            image = cv2.imread(str(scene_dir / "resized_images" / name))
            rectified = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            cv2.imwrite(str(out_image_dir / name), rectified)

            depth = self._render_depth(rasterizer, mesh, poses[name], K, distortion, height, width)
            rectified_depth = cv2.remap(
                depth, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            encode_depth_png(out_depth_dir / (Path(name).stem + ".png"), rectified_depth)
            processed += 1

        self._write_transforms(transforms, new_K, out_transforms)
        tqdm.write(f"{scene_id}: {processed} frames")

    def _load_mesh(self, scene_id: str) -> Meshes:
        mesh_path = self.data_root / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
        o3d_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        verts = torch.tensor(np.asarray(o3d_mesh.vertices), dtype=torch.float32, device=self.device)
        faces = torch.tensor(np.asarray(o3d_mesh.triangles), dtype=torch.int64, device=self.device)
        return Meshes(verts=[verts], faces=[faces])

    def _render_depth(
        self,
        rasterizer: MeshRasterizer,
        mesh: Meshes,
        pose: np.ndarray,
        K: np.ndarray,
        distortion: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Rasterize depth for one camera; returns metres."""
        cameras = make_fisheye_camera(pose, K, distortion, height, width).to(self.device)
        with torch.no_grad():
            zbuf = rasterizer(mesh, cameras=cameras).zbuf[0, :, :, 0]
        depth = zbuf.cpu().numpy()
        depth[depth == -1] = 0.0  # pixels with no mesh hit
        return depth

    @staticmethod
    def _write_transforms(transforms: dict, new_K: np.ndarray, out_path: Path) -> None:
        pinhole_fields = {
            "camera_model": "PINHOLE",
            "fl_x": float(new_K[0, 0]),
            "fl_y": float(new_K[1, 1]),
            "cx": float(new_K[0, 2]),
            "cy": float(new_K[1, 2]),
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
        }
        out = {**transforms, **pinhole_fields, "test_frames": transforms.get("test_frames", [])}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=4))
