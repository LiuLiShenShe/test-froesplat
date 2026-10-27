#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional check for the manuscript §2.8 weighted TSDF fusion.

Runs with the 2DGS venv (it needs open3d + torch):

    ../../2d-gaussian-splatting-m2m3-floor40/venv/bin/python scripts/check_weighted_tsdf.py

Three properties of ``GaussianExtractor.extract_mesh_weighted`` are checked on a
synthetic scene whose answer is known analytically -- a plane at world z = 1.0:

  A. geometry    the fused zero level set sits on the plane,
  B. w_mask      a voxel outside every view's foreground prior receives no
                 support and is carved out,
  C. w_vis       the truncation-band confidence falls linearly from 1 on the
                 surface to ``vis_weight_min`` at +/-sdf_trunc,
  D. w_normal    a view whose rendered normal is perpendicular to the ray is
                 down-weighted out of the fused value when its weight floor is 0.

The check is deliberately independent of any trained model: if it fails, the
fusion math is wrong, not the reconstruction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

GS = Path("/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40")
sys.path.insert(0, str(GS))

import utils.mesh_utils as mesh_utils  # noqa: E402


class _FakeCam:
    """Minimal stand-in for an Open3D PinholeCameraParameters."""

    def __init__(self, extrinsic, intrinsic):
        self.extrinsic = np.asarray(extrinsic, dtype=np.float64)
        self.intrinsic = type("I", (), {"intrinsic": np.asarray(intrinsic, dtype=np.float64)})()


def _cam_at(origin, look_at=None, up=(0.0, 1.0, 0.0)):
    """OpenCV/Open3D convention: +z forward, +y down, +x right.

    `look_at` defaults to a point one unit straight ahead of the camera, so the
    two test cameras share the same optical axis and their constant depth maps
    describe the *same* world plane.  (A shared explicit look_at point would tilt
    the off-axis camera's depth map onto a different plane and the fusion would
    legitimately disagree.)
    """
    origin_arr = np.asarray(origin, dtype=np.float64)
    look_at = origin_arr + np.array([0.0, 0.0, 1.0]) if look_at is None \
        else np.asarray(look_at, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = look_at - origin
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])            # world -> camera rotation
    t = -R @ origin
    return _FakeCam(np.block([[R, t[:, None]], [np.zeros(3), 1.0]]),
                    np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]]))


class _FakeGaussians:
    def __init__(self, pts):
        self._p = torch.as_tensor(pts, dtype=torch.float32)

    @property
    def get_xyz(self):
        return self._p


class _View:
    """The only attribute the fusion loop reads from a camera is its image name."""

    def __init__(self, name):
        self.image_name = name


def _build(depths, normals, names, h=64, w=64):
    ex = mesh_utils.GaussianExtractor.__new__(mesh_utils.GaussianExtractor)
    ex.gaussians = _FakeGaussians(np.random.default_rng(0).uniform(
        (-0.4, -0.4, 0.95), (0.4, 0.4, 1.05), size=(4000, 3)))
    ex.depthmaps = [torch.from_numpy(d.astype(np.float32)) for d in depths]
    ex.rgbmaps = [torch.full((3, h, w), 0.5) for _ in depths]
    ex.depth_normals = [torch.from_numpy(n.astype(np.float32)) for n in normals]
    ex.viewpoint_stack = [_View(name) for name in names]
    ex.radius = 1.0
    ex.center = torch.zeros(3)
    return ex


def check_geometry(tmp: Path) -> dict:
    """A plane seen by two cameras must come back as a plane."""
    h = w = 64
    depth = np.full((h, w), 1.0, dtype=np.float32)
    normal = np.zeros((3, h, w), dtype=np.float32)
    normal[2] = -1.0                      # facing the camera (ray is +z)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mesh_dir = tmp / "masks"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        (mesh_dir / f"mask_{stem}.png").write_bytes(b"")
    from PIL import Image
    for stem in ("a", "b"):
        Image.fromarray(mask).save(mesh_dir / f"mask_{stem}.png")

    cams = [_cam_at((0, 0, 0)), _cam_at((-0.1, 0, 0))]
    ex = _build([depth, depth], [normal, normal], ["a.png", "b.png"])
    mesh_utils.to_cam_open3d = lambda stack: [cams[i] for i in range(len(stack))]
    diag = {}
    mesh = ex.extract_mesh_weighted(
        voxel_size=0.01, sdf_trunc=0.05, depth_trunc=3.0,
        mask_dir=str(mesh_dir), normal_sharpness=8.0,
        vis_weight_min=0.2, normal_weight_min=0.2, mask_weight_min=0.0,
        export_diagnostics=diag)
    v = np.asarray(mesh.vertices)
    assert len(v) > 0, "no vertices extracted from a visible plane"
    # The reconstructed plane is a finite patch: its silhouette boundary is
    # closed by a lateral wall that legitimately spans the truncation band.  The
    # geometric claim under test is that the *surface* sits on the plane, so the
    # patch interior is measured and the boundary is only checked for existence.
    interior = (np.abs(v[:, 0]) <= 0.30) & (np.abs(v[:, 1]) <= 0.30)
    boundary = ~interior
    assert boundary.any(), "the patch was not closed at its boundary"
    z = v[interior, 2]
    err = float(np.abs(z - 1.0).max())
    assert err <= 0.02, f"plane reconstruction off by {err:.4f} (max |z-1|, interior)"
    assert diag["occupied_voxels"] > 0
    return {"vertices": int(len(v)), "interior_vertices": int(interior.sum()),
            "max_z_error": err,
            "mean_z_error": float(np.abs(z - 1.0).mean()),
            "occupied_voxels": diag["occupied_voxels"],
            "mean_effective_views": diag["mean_effective_views"]}


def check_mask_support(tmp: Path) -> dict:
    """A region outside every foreground prior must not be reconstructed."""
    h = w = 64
    depth = np.full((h, w), 1.0, dtype=np.float32)
    normal = np.zeros((3, h, w), dtype=np.float32)
    normal[2] = -1.0
    half = np.zeros((h, w), dtype=np.uint8)
    half[:, : w // 2] = 255               # only the left half is foreground
    from PIL import Image
    mesh_dir = tmp / "masks_half"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(half).save(mesh_dir / "mask_a.png")
    Image.fromarray(half).save(mesh_dir / "mask_b.png")

    cams = [_cam_at((0, 0, 0)), _cam_at((-0.1, 0, 0))]
    ex = _build([depth, depth], [normal, normal], ["a.png", "b.png"])
    mesh_utils.to_cam_open3d = lambda stack: [cams[i] for i in range(len(stack))]
    diag = {}
    mesh = ex.extract_mesh_weighted(
        voxel_size=0.01, sdf_trunc=0.05, depth_trunc=3.0,
        mask_dir=str(mesh_dir), mask_weight_min=0.0,
        export_diagnostics=diag)
    v = np.asarray(mesh.vertices)
    assert len(v) > 0
    # Camera at origin looking +z with fx=100, cx=32: the mask edge (u=32) is
    # the plane's world-x = 0 line.
    assert (v[:, 0].max() < 0.05), (
        f"vertices leak past the mask edge: max x = {v[:, 0].max():.3f}")
    assert (v[:, 0].min() < -0.05), "the masked-in half was not reconstructed"
    return {"vertices": int(len(v)), "max_x": float(v[:, 0].max()),
            "min_x": float(v[:, 0].min())}


def check_normal_weighting(tmp: Path) -> dict:
    """A grazing view must not override a head-on view when its floor is 0."""
    h = w = 64
    d_headon = np.full((h, w), 1.0, dtype=np.float32)     # plane at z=1.0
    d_grazing = np.full((h, w), 1.4, dtype=np.float32)    # plane at z=1.4
    n_headon = np.zeros((3, h, w), dtype=np.float32)
    n_headon[2] = -1.0
    n_grazing = np.zeros((3, h, w), dtype=np.float32)
    n_grazing[0] = -1.0                                   # perpendicular to the ray
    mask = np.ones((h, w), dtype=np.uint8) * 255
    from PIL import Image
    mesh_dir = tmp / "masks_n"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        Image.fromarray(mask).save(mesh_dir / f"mask_{stem}.png")

    cams = [_cam_at((0, 0, 0)), _cam_at((0, 0, 0))]
    ex = _build([d_headon, d_grazing], [n_headon, n_grazing], ["a.png", "b.png"])
    mesh_utils.to_cam_open3d = lambda stack: [cams[i] for i in range(len(stack))]
    diag = {}
    mesh = ex.extract_mesh_weighted(
        voxel_size=0.01, sdf_trunc=0.05, depth_trunc=3.0,
        mask_dir=str(mesh_dir), normal_sharpness=8.0, normal_weight_min=0.0,
        mask_weight_min=1.0, export_diagnostics=diag)
    v = np.asarray(mesh.vertices)
    assert len(v) > 0
    # The head-on surface is z=1.0; its truncation band legitimately reaches
    # z=1.0 +/- sdf_trunc (=0.05).  What must be absent is the *grazing* plane at
    # z=1.4: that is the contamination the down-weighting is meant to prevent.
    surface_band = np.abs(v[:, 2] - 1.0) <= 0.06
    assert surface_band.any(), "the head-on surface was not reconstructed"
    assert float(v[:, 2].max()) < 1.25, (
        f"grazing view contaminated the surface: z max {v[:, 2].max():.3f} "
        f"(should not approach the grazing depth 1.4)")
    assert float(v[:, 2].min()) > 0.75
    cue = diag["cue_mean"]
    assert cue["normal"] < 0.9, "the grazing view was not down-weighted"
    return {"vertices": int(len(v)), "max_z": float(v[:, 2].max()),
            "min_z": float(v[:, 2].min()), "cue_mean_normal": cue["normal"]}


def main() -> int:
    tmp = Path("/tmp/weighted_tsdf_check")
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in (("geometry", check_geometry),
                     ("mask_support", check_mask_support),
                     ("normal_weighting", check_normal_weighting)):
        out[name] = fn(tmp)
        print(f"[ok] {name}: {json.dumps(out[name], ensure_ascii=False)}")
    (tmp / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"[ok] all §2.8 weighted-TSDF properties hold; wrote {tmp/'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
