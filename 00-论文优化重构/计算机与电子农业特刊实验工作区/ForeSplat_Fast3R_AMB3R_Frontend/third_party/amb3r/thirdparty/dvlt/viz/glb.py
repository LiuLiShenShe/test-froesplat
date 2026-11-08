# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""GLB export helpers for point clouds with an optional camera-path overlay.

The exported scene contains up to two nodes:

* ``points`` — a ``trimesh.PointCloud`` of the supplied vertices + colours.
* ``camera_overlay`` — the combined rainbow frustum / wireframe / trajectory
  overlay (see :mod:`dvlt.viz.camera_overlay`). Tagged with
  ``KHR_materials_unlit`` so vertex colours render uniformly across viewers
  (model-viewer, MeshLab, Blender, three.js, …).

Coordinate convention: input points and ``cameras_c2w`` are expected in
OpenCV (y-down, z-forward, right-handed) frame — the convention produced by
dvlt's prediction pipeline. The scene is rotated 180° around X on export so
the GLB lands in the OpenGL / glTF (y-up, z-back) frame that web viewers
expect.
"""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from dvlt.viz.camera_overlay import (
    combined_overlay_mesh,
    patch_glb_material_unlit,
    scene_scale,
)


__all__ = [
    "opencv_to_opengl_transform",
    "pointcloud_to_glb",
]


def opencv_to_opengl_transform() -> np.ndarray:
    """180° rotation around X mapping OpenCV (y-down, z-fwd) → glTF (y-up, z-back)."""
    m = np.eye(4)
    m[1, 1] = -1
    m[2, 2] = -1
    return m


def pointcloud_to_glb(
    pts: np.ndarray,
    rgb: np.ndarray,
    output_path: Optional[Path] = None,
    *,
    name: Optional[str] = None,
    cameras_c2w: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    image_hw: Optional[tuple[int, int]] = None,
    frustum_frac: float = 0.01,
    edge_frac: float = 0.04,
    line_frac: float = 0.0002,
) -> str:
    """Export a coloured point cloud (and optional camera path) as a GLB file.

    Args:
      pts: ``(P, 3)`` float vertex positions in OpenCV frame.
      rgb: ``(P, 3)`` uint8 colours.
      output_path: target GLB path; if ``None``, a ``tempfile.mktemp(".glb")``
        path is used (Gradio's expected hand-off).
      name: optional scene/material display name. Currently unused by the
        exporter beyond logging, but accepted so downstream callers can label
        the file consistently with their per-sequence name.
      cameras_c2w: ``(N, 4, 4)`` camera-to-world matrices in OpenCV frame. When
        provided and ``frustum_frac > 0``, a rainbow frustum + trajectory
        overlay is added to the scene as the ``camera_overlay`` node.
      intrinsics: ``(N, 3, 3)`` per-camera intrinsics. Optional — when omitted,
        a synthetic 4:3 / 60°-HFoV pin-hole is used so the trajectory still
        reads even when intrinsics aren't readily available.
      image_hw: ``(H, W)`` source image size for back-projecting frustum
        corners. Optional alongside ``intrinsics`` (same default fallback).
      frustum_frac: pyramid depth as a fraction of the combined scene scale.
        Defaults to 1% — perceptually consistent across very different scene
        scales. ``frustum_frac <= 0`` disables the overlay.
      edge_frac / line_frac: forwarded to
        :func:`dvlt.viz.camera_overlay.combined_overlay_mesh`; pass these if
        the caller wants finer control over wireframe vs. trajectory radii.

    Returns:
      Absolute path to the written GLB.
    """
    rgba = np.c_[rgb, np.full(len(rgb), 255, dtype=np.uint8)]
    cloud = trimesh.PointCloud(vertices=pts, colors=rgba)
    scene = trimesh.Scene()
    scene.add_geometry(cloud, node_name="points")

    overlay_added = False
    if cameras_c2w is not None and frustum_frac > 0.0:
        c2ws = np.asarray(cameras_c2w, dtype=np.float64)
        if c2ws.ndim != 3 or c2ws.shape[-2:] != (4, 4):
            raise ValueError(f"cameras_c2w must be (N, 4, 4); got {c2ws.shape}")
        K = np.asarray(intrinsics, dtype=np.float64) if intrinsics is not None else None
        ss = scene_scale(pts, c2ws[:, :3, 3])
        overlay = combined_overlay_mesh(
            c2ws,
            ss,
            intrinsics=K,
            image_hw=image_hw,
            frustum_frac=frustum_frac,
            edge_frac=edge_frac,
            line_frac=line_frac,
        )
        if overlay is not None:
            scene.add_geometry(overlay, node_name="camera_overlay")
            overlay_added = True

    scene.apply_transform(opencv_to_opengl_transform())

    if output_path is None:
        path = tempfile.mktemp(suffix=".glb")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        path = str(output_path)
    scene.export(file_obj=path)
    if overlay_added:
        patch_glb_material_unlit(path, material_name="camera_overlay")
    _ = name  # accepted for caller-side parity; reserved for future use
    return path
