"""Run the real GGGS readers + Gaussian initialisation on an exported dataset.

This exercises the same code path ForeSplat uses in training
(``scene.dataset_readers.readColmapSceneInfo`` + ``fetchPly`` +
``GaussianModel.create_from_pcd``) to prove the dataset is consumable, without
running optimisation.  ``simple_knn`` is needed for ``create_from_pcd``; when
it is unavailable we fall back to a *read-only* verification.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys


def _preload_cuda_ext_deps() -> bool:
    """Preload the shared libraries ``simple_knn._C`` needs, globally.

    The pip-installed ``simple_knn`` extension links against ``libc10.so``
    (shipped inside the torch package) and pulls in ``libcusparse.so`` whose
    symbols live in the pip ``nvidia-nvjitlink`` package.  Neither directory
    is on the default loader path, so a plain ``import simple_knn`` fails.
    Runtime edits to ``LD_LIBRARY_PATH`` do not affect ``dlopen``, so the
    libraries are loaded explicitly with ``RTLD_GLOBAL`` instead.
    """
    try:
        import simple_knn._C  # noqa: F401

        return True
    except ImportError:
        pass

    import torch

    loaded = False
    torch_dir = os.path.dirname(torch.__file__)
    site_dir = os.path.dirname(os.path.dirname(torch_dir))
    for pattern in (
        os.path.join(site_dir, "nvidia", "nvjitlink", "lib", "libnvJitLink.so*"),
        os.path.join(torch_dir, "lib", "libc10.so"),
    ):
        hit = sorted(glob.glob(pattern))
        if hit:
            ctypes.CDLL(hit[-1], mode=ctypes.RTLD_GLOBAL)
            loaded = True
    return loaded


def check_dataset_and_init_gaussians(dataset_dir: str, gggs_repo: str) -> dict:
    """Return a status dict for the batch report."""
    status: dict = {
        "dataset_dir": dataset_dir,
        "gggs_repo": gggs_repo,
        "colmap_read": "not attempted",
        "ply_points": None,
        "camera_count": None,
        "gaussian_init": "not attempted",
        "gaussian_shapes": None,
        "error": None,
    }
    if gggs_repo not in sys.path:
        sys.path.insert(0, gggs_repo)

    preload_ok = _preload_cuda_ext_deps()
    try:
        # Reuse the loader from verify.py: it stubs optional dependencies
        # (open3d, simple_knn) that GGGS imports unconditionally but the
        # read-only path does not need.
        from .verify import _load_gggs_dataset_readers

        dataset_readers, _ = _load_gggs_dataset_readers(gggs_repo)
        readColmapSceneInfo = dataset_readers.readColmapSceneInfo
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"GGGS import failed: {exc}"
        return status

    try:
        scene_info = readColmapSceneInfo(dataset_dir, None, False)
        status["colmap_read"] = "ok"
        pcd = scene_info.point_cloud
        if pcd is None:
            status["error"] = "GGGS point cloud is None (points3D.ply unreadable)"
            return status
        status["ply_points"] = int(pcd.points.shape[0])
        status["ply_has_normals"] = bool(getattr(pcd, "normals", None) is not None)
        status["camera_count"] = len(scene_info.train_cameras) + len(scene_info.test_cameras)
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"readColmapSceneInfo failed: {exc}"
        return status

    try:
        from scene.gaussian_model import GaussianModel  # type: ignore

        gaussian = GaussianModel(3, 3)
        spatial_lr_scale = 1.0
        gaussian.create_from_pcd(pcd, spatial_lr_scale)
        status["gaussian_init"] = "ok"
        status["gaussian_shapes"] = {
            "xyz": list(gaussian.get_xyz.shape),
            "features": list(gaussian.get_features.shape),
        }
        del gaussian
    except Exception as exc:  # noqa: BLE001
        status["gaussian_init"] = f"failed: {type(exc).__name__}: {exc}"
        if not preload_ok:
            status["gaussian_init"] += " (simple_knn CUDA extension unavailable)"
    return status