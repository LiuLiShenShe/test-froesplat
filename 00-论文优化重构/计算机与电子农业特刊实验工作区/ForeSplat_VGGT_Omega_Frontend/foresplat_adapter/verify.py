"""Verify that the exported dataset can be read by the GGGS dataset reader.

Imports ``scene.dataset_readers.readColmapSceneInfo`` from a local clone of
https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids and
runs the exact code path GGGS uses when loading a COLMAP dataset
(``readColmapCameras`` -> ``CameraInfo`` -> ``utils.camera_utils.loadCam``).

``open3d`` is only needed by GGGS's ``fetchOpen3DPly`` fallback (which we do
not exercise), so it is stubbed when not installed.
"""

from __future__ import annotations

import os
import sys
import types


def _install_stub(name: str, attributes: dict | None = None) -> None:
    """Create an importable stub module for a missing optional dependency."""
    module = types.ModuleType(name)
    for attr, value in (attributes or {}).items():
        setattr(module, attr, value)
    sys.modules.setdefault(name, module)


def _load_gggs_dataset_readers(gggs_repo: str):
    gggs_repo = os.path.abspath(gggs_repo)
    if not os.path.isdir(os.path.join(gggs_repo, "scene")):
        raise FileNotFoundError(f"GGGS repository not found at {gggs_repo}")
    if gggs_repo not in sys.path:
        sys.path.insert(0, gggs_repo)

    # GGGS's read path pulls in scene.gaussian_model (for BasicPointCloud) and
    # scene.dataset_readers (open3d). The optional CUDA extensions (simple_knn)
    # and open3d are only needed for the training path, so stub them when absent.
    for name, attributes in (
        ("simple_knn", {}),
        ("simple_knn._C", {"distCUDA2": lambda *a, **k: None}),
        ("open3d", None),
    ):
        try:
            __import__(name)
        except Exception:
            _install_stub(name, attributes)
    if "open3d" in sys.modules and not hasattr(sys.modules["open3d"], "io"):
        sys.modules["open3d"].io = types.SimpleNamespace(read_point_cloud=lambda *a, **k: None)

    from scene import dataset_readers  # noqa: PLC0415
    from utils import camera_utils  # noqa: PLC0415

    return dataset_readers, camera_utils


def verify_dataset_static(dataset_dir: str) -> dict:
    """Header-only dataset checks (no full image decode, no GGGS import).

    Verifies the things the GGGS reader would reject: camera files present,
    PINHOLE cameras, camera size == image size, masks present and image names
    consistent.  Used by the batch runner instead of loading every image.
    """
    from PIL import Image

    dataset_dir = os.path.abspath(dataset_dir)
    sparse = os.path.join(dataset_dir, "sparse", "0")
    report = {"checks": {}, "cameras": 0, "images": 0}

    for name in ("cameras.txt", "images.txt"):
        report["checks"][f"{name}_exists"] = os.path.isfile(os.path.join(sparse, name))
    report["checks"]["points3D_ply_exists"] = os.path.isfile(os.path.join(sparse, "points3D.ply"))
    if not report["checks"]["cameras.txt_exists"] or not report["checks"]["images.txt_exists"]:
        report["all_checks_passed"] = False
        return report

    camera_sizes: dict[int, tuple[int, int]] = {}
    models = set()
    with open(os.path.join(sparse, "cameras.txt"), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            camera_sizes[int(parts[0])] = (int(parts[2]), int(parts[3]))
            models.add(parts[1])
    report["checks"]["single_pinhole_model"] = models <= {"PINHOLE", "SIMPLE_PINHOLE"}
    report["cameras"] = len(camera_sizes)

    images: list[tuple[str, int]] = []
    with open(os.path.join(sparse, "images.txt"), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # COLMAP images.txt is two lines per image: the image line has
            # exactly IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (10 tokens);
            # the following POINTS2D observation line has any other count and
            # must be skipped (its last token is a POINT3D_ID, not a filename).
            if len(parts) != 10:
                continue
            images.append((parts[9], int(parts[8])))
    report["images"] = len(images)

    size_ok = True
    masks_ok = True
    missing = []
    for name, camera_id in images:
        path = os.path.join(dataset_dir, "images", os.path.basename(name))
        if not os.path.isfile(path):
            missing.append(name)
            size_ok = False
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:  # noqa: BLE001
            size_ok = False
            continue
        if camera_sizes.get(camera_id) != (width, height):
            size_ok = False
        mask_path = os.path.join(dataset_dir, "masks", os.path.splitext(os.path.basename(name))[0] + ".png")
        if not os.path.isfile(mask_path):
            masks_ok = False

    report["checks"]["all_images_present"] = not missing
    report["checks"]["image_size_matches_intrinsics"] = size_ok
    report["checks"]["all_masks_present"] = masks_ok
    report["checks"]["camera_count_positive"] = len(camera_sizes) > 0
    report["all_checks_passed"] = all(v is True for v in report["checks"].values())
    return report


def verify_gggs_dataset(dataset_dir: str, gggs_repo: str, resolution: int = 1) -> dict:
    """Run the GGGS COLMAP loading path on ``dataset_dir`` and report results."""
    dataset_readers, camera_utils = _load_gggs_dataset_readers(gggs_repo)

    scene_info = dataset_readers.readColmapSceneInfo(os.path.abspath(dataset_dir), None, eval=False)

    args = types.SimpleNamespace(data_device="cpu", resolution=resolution)
    cameras = camera_utils.cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)

    report = {
        "gggs_repo": os.path.abspath(gggs_repo),
        "dataset_dir": os.path.abspath(dataset_dir),
        "num_train_cameras": len(scene_info.train_cameras),
        "num_test_cameras": len(scene_info.test_cameras),
        "ply_path": scene_info.ply_path,
        "point_cloud": None,
        "nerf_normalization_radius": float(scene_info.nerf_normalization["radius"]),
        "cameras": [],
        "checks": {},
    }

    pcd = scene_info.point_cloud
    if pcd is None:
        raise RuntimeError("GGGS failed to load the initialisation point cloud (points3D.ply)")
    report["point_cloud"] = {
        "num_points": int(pcd.points.shape[0]),
        "color_range": [float(pcd.colors.min()), float(pcd.colors.max())] if len(pcd.colors) else None,
        "finite_points": bool(pcd.points.all()),
    }

    for cam_info, cam in zip(scene_info.train_cameras, cameras):
        report["cameras"].append(
            {
                "image_name": cam.image_name,
                "colmap_camera_width": int(cam_info.width),
                "colmap_camera_height": int(cam_info.height),
                "image_width": int(cam.image_width),
                "image_height": int(cam.image_height),
                "FoVx": float(cam.FoVx),
                "FoVy": float(cam.FoVy),
                "Fx": float(cam.Fx),
                "Fy": float(cam.Fy),
                "alpha_mask_set": cam.gt_mask is not None,
            }
        )

    # Consistency checks -----------------------------------------------------
    checks = report["checks"]
    checks["ply_exists"] = os.path.isfile(scene_info.ply_path)
    checks["num_points_positive"] = report["point_cloud"]["num_points"] > 0
    checks["camera_count_matches"] = len(cameras) == len(scene_info.train_cameras)
    sizes_ok = all(
        c["colmap_camera_width"] == c["image_width"] and c["colmap_camera_height"] == c["image_height"]
        for c in report["cameras"]
    )
    checks["image_size_matches_intrinsics"] = sizes_ok
    masks_dir = os.path.join(os.path.abspath(dataset_dir), "masks")
    checks["masks_dir_exists"] = os.path.isdir(masks_dir)
    expected_masks = [os.path.splitext(os.path.basename(c["image_name"]))[0] + ".png" for c in report["cameras"]]
    checks["all_masks_present"] = all(os.path.isfile(os.path.join(masks_dir, m)) for m in expected_masks)
    checks["colmap_files_present"] = all(
        os.path.isfile(os.path.join(os.path.abspath(dataset_dir), "sparse", "0", f))
        for f in ("cameras.bin", "images.bin", "points3D.bin")
    )
    report["all_checks_passed"] = all(v is True for k, v in checks.items())
    return report
