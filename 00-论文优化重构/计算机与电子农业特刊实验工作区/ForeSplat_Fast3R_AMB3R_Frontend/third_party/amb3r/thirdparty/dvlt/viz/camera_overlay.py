# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Trimesh builders for the rainbow camera-path overlay (frustums + lines).

Used by :mod:`dvlt.scripts.gradio_demo` to add a single combined overlay to
a GLB. Exposed as three independent mesh builders (pyramids, edges, line)
plus :func:`combined_overlay_mesh` so callers that want per-layer materials
or alphas can mix-and-match.

The frustum at frame *i* is a 5-vertex pyramid whose apex sits at the camera
origin and whose base corners are the four image-plane corners back-projected
to a camera-space depth ``depth = frustum_frac * scene_scale``. With explicit
``intrinsics`` and ``image_hw`` the base matches the model's view frustum
exactly; otherwise a synthetic 4:3 / 60°-HFoV pin-hole is used so the
trajectory still reads when intrinsics aren't handy.

All meshes use per-vertex ``COLOR_0`` RGBA so every viewer — MeshLab,
model-viewer, three.js, Blender — picks up the rainbow colours, and any
combined-overlay path tags the GLB material as ``KHR_materials_unlit`` so
viewers that don't honour ``COLOR_0`` on ``TRIANGLES`` under PBR (notably
MeshLab) still render the colours instead of a flat white.

Geometry sizing:
- ``frustum_frac``: pyramid depth as a fraction of the scene scale (~1% works
  well across very different scales).
- ``edge_frac``: edge-cylinder radius as a fraction of the **pyramid depth**
  (~4% by default; couples wireframe thickness to the slider).
- ``line_frac``: trajectory-cylinder radius as a fraction of the **scene
  scale** (~0.02% by default; thinner than the frustum edges so the path
  doesn't overpower the frustums).
"""

import json
import struct
from typing import Optional

import numpy as np
import trimesh


__all__ = [
    "scene_scale",
    "rainbow_colors",
    "camera_frustums_mesh",
    "camera_frustum_edges_mesh",
    "camera_lines_mesh",
    "combined_overlay_mesh",
    "patch_glb_material_unlit",
]


# ---------------------------------------------------------------------------
# Defaults for the synthetic frustum used when no intrinsics are supplied
# ---------------------------------------------------------------------------

# 4:3 / 60° HFoV — small, recognizable frustum shape that doesn't look weirdly
# squat or stretched. Used purely as a visual default; never passes through
# the model's geometry pipeline.
_DEFAULT_IMAGE_HW: tuple[int, int] = (3, 4)
_DEFAULT_HFOV_DEG: float = 60.0


def _default_intrinsics(image_hw: tuple[int, int], hfov_deg: float = _DEFAULT_HFOV_DEG) -> np.ndarray:
    """Synthetic ``K`` for ``image_hw`` with horizontal FoV ``hfov_deg``.

    Square pixels (``fx == fy``), principal point at image centre. Returned
    shape is ``(3, 3)``; broadcast to ``(N, 3, 3)`` at the call site.
    """
    H, W = image_hw
    fx = (float(W) * 0.5) / np.tan(np.deg2rad(hfov_deg) * 0.5)
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fx
    K[0, 2] = float(W) * 0.5
    K[1, 2] = float(H) * 0.5
    return K


def _resolve_intrinsics(
    intrinsics: Optional[np.ndarray],
    image_hw: Optional[tuple[int, int]],
    n_cameras: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return ``(intrinsics_NxK, image_hw)``, filling defaults when missing.

    ``intrinsics`` may be ``(3, 3)`` (broadcast to N) or ``(N, 3, 3)``.
    """
    if image_hw is None:
        image_hw = _DEFAULT_IMAGE_HW
    if intrinsics is None:
        K = _default_intrinsics(image_hw)
        return np.broadcast_to(K, (n_cameras, 3, 3)).copy(), tuple(image_hw)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim == 2 and intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics, (n_cameras, 3, 3)).copy()
    if intrinsics.shape != (n_cameras, 3, 3):
        raise ValueError(f"intrinsics must be (3, 3) or (N=={n_cameras}, 3, 3); got {intrinsics.shape}")
    return intrinsics, tuple(image_hw)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def scene_scale(pts: Optional[np.ndarray], cam_centers: np.ndarray) -> float:
    """Robust scene scale for sizing camera overlays.

    Uses the 99th-percentile distance from the median of the union of point
    cloud vertices and camera centres. Median + percentile makes this stable
    against the long tails that depth predictions sometimes produce.
    """
    parts = [a for a in (pts, cam_centers) if a is not None and len(a) > 0]
    if not parts:
        return 1.0
    all_pts = np.concatenate(parts, axis=0)
    centroid = np.median(all_pts, axis=0)
    dists = np.linalg.norm(all_pts - centroid, axis=1)
    if dists.size == 0:
        return 1.0
    return max(float(np.quantile(dists, 0.99)), 1e-6)


def rainbow_colors(n: int, cmap: str = "turbo") -> np.ndarray:
    """``n × 3`` uint8 RGB sampled from a matplotlib colormap on ``[0, 1]``.

    ``turbo`` is the modern perceptually-uniform rainbow: ``t=0`` is
    blue-purple and ``t=1`` is dark red, so the trajectory direction reads
    at a glance. Matplotlib is imported lazily so import-only paths (e.g.
    Gradio app boot without inference) don't pay the cost.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    import matplotlib.cm as mpl_cm  # noqa: PLC0415 — lazy

    if n == 1:
        ts = np.array([0.5])
    else:
        ts = np.linspace(0.0, 1.0, n)
    rgba = getattr(mpl_cm, cmap)(ts)
    return (rgba[:, :3] * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal mesh building blocks
# ---------------------------------------------------------------------------


def _set_vertex_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
    """Paint every vertex of ``mesh`` with the same RGBA uint8 colour.

    Per-vertex ``COLOR_0`` is the GLTF attribute that *every* viewer
    (MeshLab, model-viewer, three.js, Blender) reads. ``face_colors`` /
    PBR ``baseColorFactor`` is also spec-compliant but MeshLab in particular
    ignores it and renders meshes white, so we route everything through
    vertex colours for portability.
    """
    n = len(mesh.vertices)
    mesh.visual = trimesh.visual.color.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(rgba.astype(np.uint8).reshape(1, 4), (n, 1)),
    )


def _frustum_corners_world(c2w: np.ndarray, K: np.ndarray, image_hw: tuple[int, int], depth: float) -> np.ndarray:
    """Return ``(5, 3)`` world-space vertices: ``[apex, c00, c10, c11, c01]``.

    Corners are at the four image-plane pixel corners back-projected to
    camera-space depth ``depth``, then transformed to world via ``c2w``.
    Order is CCW when viewed from the apex (which is conventional for
    pyramid winding so the side normals face outward).
    """
    H, W = image_hw
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    corners_px = np.array([[0.0, 0.0], [W, 0.0], [W, H], [0.0, H]], dtype=np.float64)
    rays = np.stack(
        [(corners_px[:, 0] - cx) / fx, (corners_px[:, 1] - cy) / fy, np.ones(4)],
        axis=-1,
    )
    corners_cam = rays * float(depth)  # (4, 3)
    verts_cam = np.vstack([np.zeros((1, 3)), corners_cam])  # (5, 3)
    verts_h = np.c_[verts_cam, np.ones(5)]
    return (c2w @ verts_h.T).T[:, :3]


# Pyramid triangles, double-sided (each face appears twice with reversed
# winding) so transparent frustums render from any vantage even when the
# default-back-face-culling glTF / three.js / model-viewer pipeline applies.
_PYRAMID_FACES = np.array(
    [
        [0, 1, 2],  # side: apex - c00 - c10
        [0, 2, 3],  # side: apex - c10 - c11
        [0, 3, 4],  # side: apex - c11 - c01
        [0, 4, 1],  # side: apex - c01 - c00
        [1, 2, 3],  # base 1
        [1, 3, 4],  # base 2
    ],
    dtype=np.int64,
)
_PYRAMID_FACES_DS = np.concatenate([_PYRAMID_FACES, _PYRAMID_FACES[:, ::-1]], axis=0)

# Edges around the frustum: 4 apex→corner + 4 base loop.
_PYRAMID_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
]


def _stride_index(n: int, stride: int) -> np.ndarray:
    """Indices we keep for a ``stride > 1`` overlay.

    Always retains the first and last camera so the path's endpoints are
    visible regardless of stride; intermediate indices follow ``range(0, n,
    stride)``.
    """
    s = max(int(stride), 1)
    keep = sorted({0, n - 1, *range(0, n, s)})
    return np.asarray(keep, dtype=np.int64)


def _maybe_resolve_colors(colors_rgb: Optional[np.ndarray], n: int) -> np.ndarray:
    """Either accept supplied per-camera colours or sample the turbo ramp."""
    if colors_rgb is None:
        return rainbow_colors(n)
    arr = np.asarray(colors_rgb)
    if arr.shape != (n, 3):
        raise ValueError(f"colors_rgb must be (N={n}, 3); got {arr.shape}")
    return arr.astype(np.uint8)


# ---------------------------------------------------------------------------
# Public mesh builders
# ---------------------------------------------------------------------------


def camera_frustums_mesh(
    cameras_c2w: np.ndarray,
    scene_scale_abs: float,
    *,
    intrinsics: Optional[np.ndarray] = None,
    image_hw: Optional[tuple[int, int]] = None,
    stride: int = 1,
    frustum_frac: float = 0.01,
    face_alpha: int = 90,
    colors_rgb: Optional[np.ndarray] = None,
) -> Optional[trimesh.Trimesh]:
    """One concatenated mesh of all frustum pyramids (no edges, no lines).

    Pyramid faces carry the camera's rainbow colour with alpha
    ``face_alpha`` (0–255) per vertex so a downstream ``alphaMode=BLEND``
    material renders them translucent.

    Returns ``None`` when ``cameras_c2w`` is empty or ``frustum_frac <= 0``.
    """
    cameras_c2w = np.asarray(cameras_c2w, dtype=np.float64)
    n = len(cameras_c2w)
    if n == 0 or frustum_frac <= 0.0:
        return None
    intr, hw = _resolve_intrinsics(intrinsics, image_hw, n)
    keep = _stride_index(n, stride)
    colors = _maybe_resolve_colors(colors_rgb, n)
    depth = float(frustum_frac) * float(scene_scale_abs)

    pieces: list[trimesh.Trimesh] = []
    for i in keep:
        V = _frustum_corners_world(cameras_c2w[i], intr[i], hw, depth)
        pyr = trimesh.Trimesh(vertices=V, faces=_PYRAMID_FACES_DS, process=False)
        _set_vertex_color(pyr, np.array([colors[i, 0], colors[i, 1], colors[i, 2], int(face_alpha)]))
        pieces.append(pyr)
    return trimesh.util.concatenate(pieces) if pieces else None


def camera_frustum_edges_mesh(
    cameras_c2w: np.ndarray,
    scene_scale_abs: float,
    *,
    intrinsics: Optional[np.ndarray] = None,
    image_hw: Optional[tuple[int, int]] = None,
    stride: int = 1,
    frustum_frac: float = 0.01,
    edge_frac: float = 0.04,
    darken: float = 1.0,
    colors_rgb: Optional[np.ndarray] = None,
) -> Optional[trimesh.Trimesh]:
    """One concatenated mesh of frustum-wireframe edge cylinders (always opaque).

    Each frustum contributes 8 edges (4 apex-to-corner + 4 base-loop), built
    as thin cylinders so the lines have a visible thickness in viewers that
    rasterize glTF ``LINES`` at 1px. ``darken`` multiplies the camera colour
    (use < 1 to make the edges read as a slightly darker outline against the
    transparent face).
    """
    cameras_c2w = np.asarray(cameras_c2w, dtype=np.float64)
    n = len(cameras_c2w)
    if n == 0 or frustum_frac <= 0.0 or edge_frac <= 0.0:
        return None
    intr, hw = _resolve_intrinsics(intrinsics, image_hw, n)
    keep = _stride_index(n, stride)
    colors = _maybe_resolve_colors(colors_rgb, n)
    depth = float(frustum_frac) * float(scene_scale_abs)
    edge_radius = float(edge_frac) * depth
    if edge_radius <= 0.0:
        return None

    pieces: list[trimesh.Trimesh] = []
    for i in keep:
        V = _frustum_corners_world(cameras_c2w[i], intr[i], hw, depth)
        tinted = (colors[i].astype(np.float32) * float(darken)).clip(0, 255).astype(np.uint8)
        edge_color = np.array([tinted[0], tinted[1], tinted[2], 255], dtype=np.uint8)
        for a, b in _PYRAMID_EDGES:
            seg = np.stack([V[a], V[b]], axis=0)
            if np.linalg.norm(seg[1] - seg[0]) < 1e-9:
                continue
            cyl = trimesh.creation.cylinder(radius=edge_radius, segment=seg, sections=6)
            _set_vertex_color(cyl, edge_color)
            pieces.append(cyl)
    return trimesh.util.concatenate(pieces) if pieces else None


def camera_lines_mesh(
    cameras_c2w: np.ndarray,
    scene_scale_abs: float,
    *,
    stride: int = 1,
    line_frac: float = 0.0002,
    colors_rgb: Optional[np.ndarray] = None,
) -> Optional[trimesh.Trimesh]:
    """One concatenated cylinder polyline through the (strided) camera centres.

    Segment colours are the average of the two endpoint cameras so a smooth
    rainbow gradient connects the path. ``line_frac`` is the cylinder radius
    as a fraction of ``scene_scale_abs`` — keep it well under the edge
    radius so the line reads as a thinner trace than the frustum wireframe.
    """
    cameras_c2w = np.asarray(cameras_c2w, dtype=np.float64)
    n = len(cameras_c2w)
    if n < 2 or line_frac <= 0.0:
        return None
    keep = _stride_index(n, stride)
    if len(keep) < 2:
        return None
    cam_centers = cameras_c2w[keep, :3, 3]
    colors = _maybe_resolve_colors(colors_rgb, n)[keep]
    radius = float(line_frac) * float(scene_scale_abs)
    if radius <= 0.0:
        return None

    pieces: list[trimesh.Trimesh] = []
    for j in range(len(keep) - 1):
        seg = np.stack([cam_centers[j], cam_centers[j + 1]], axis=0)
        if np.linalg.norm(seg[1] - seg[0]) < 1e-9:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, segment=seg, sections=6)
        avg = ((colors[j].astype(np.int32) + colors[j + 1].astype(np.int32)) // 2).astype(np.uint8)
        _set_vertex_color(cyl, np.array([avg[0], avg[1], avg[2], 255], dtype=np.uint8))
        pieces.append(cyl)
    return trimesh.util.concatenate(pieces) if pieces else None


def combined_overlay_mesh(
    cameras_c2w: np.ndarray,
    scene_scale_abs: float,
    *,
    intrinsics: Optional[np.ndarray] = None,
    image_hw: Optional[tuple[int, int]] = None,
    frustum_frac: float = 0.01,
    edge_frac: float = 0.04,
    line_frac: float = 0.0002,
    face_alpha: int = 90,
    edge_darken: float = 1.0,
    colors_rgb: Optional[np.ndarray] = None,
    material_name: str = "camera_overlay",
) -> Optional[trimesh.Trimesh]:
    """Single concatenated Trimesh of all three overlay layers.

    Pyramid faces (per-vertex alpha ``face_alpha``), opaque edges, and the
    trajectory polyline are joined into one mesh and tagged with a
    ``BLEND`` + ``doubleSided`` PBR material. Pair with
    :func:`patch_glb_material_unlit` after export to also get
    ``KHR_materials_unlit`` on the material — without it, viewers like
    MeshLab fall back to ``baseColorFactor`` (white) for triangles instead
    of using ``COLOR_0``.

    Used by :func:`dvlt.viz.glb.pointcloud_to_glb` to provide a single
    toggleable scene node containing the entire camera path.
    """
    cameras_c2w = np.asarray(cameras_c2w, dtype=np.float64)
    n = len(cameras_c2w)
    if n == 0 or frustum_frac <= 0.0:
        return None
    colors = _maybe_resolve_colors(colors_rgb, n)

    pieces: list[trimesh.Trimesh] = []
    frustums = camera_frustums_mesh(
        cameras_c2w,
        scene_scale_abs,
        intrinsics=intrinsics,
        image_hw=image_hw,
        stride=1,
        frustum_frac=frustum_frac,
        face_alpha=face_alpha,
        colors_rgb=colors,
    )
    edges = camera_frustum_edges_mesh(
        cameras_c2w,
        scene_scale_abs,
        intrinsics=intrinsics,
        image_hw=image_hw,
        stride=1,
        frustum_frac=frustum_frac,
        edge_frac=edge_frac,
        darken=edge_darken,
        colors_rgb=colors,
    )
    lines = camera_lines_mesh(
        cameras_c2w,
        scene_scale_abs,
        stride=1,
        line_frac=line_frac,
        colors_rgb=colors,
    )
    for piece in (frustums, edges, lines):
        if piece is not None:
            pieces.append(piece)
    if not pieces:
        return None
    overlay = trimesh.util.concatenate(pieces)
    overlay.visual.material = trimesh.visual.material.PBRMaterial(
        name=material_name,
        alphaMode="BLEND",
        doubleSided=True,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return overlay


# ---------------------------------------------------------------------------
# GLB post-processing (unlit extension)
# ---------------------------------------------------------------------------


def patch_glb_material_unlit(path: str, material_name: str = "camera_overlay") -> None:
    """Add ``KHR_materials_unlit`` to the named material in an existing GLB.

    Some viewers (notably MeshLab; some older Blender / three-rs paths)
    honour the ``COLOR_0`` vertex attribute only for ``POINTS`` primitives,
    falling back to ``baseColorFactor`` for ``TRIANGLES`` and rendering
    vertex-coloured triangle meshes as a flat ``baseColorFactor`` (white in
    our case). Marking the material as ``unlit`` flips the renderer into a
    path that multiplies ``COLOR_0`` by ``baseColorFactor`` directly with no
    lighting / PBR — the same path used for points — so colours show up
    everywhere. Other viewers (model-viewer, Babylon, latest Blender)
    implement the extension natively and honour it without altering the
    visual look (we don't need PBR shading on a thin overlay anyway).

    Edits the GLB in place via a chunked-binary-aware JSON rewrite.
    ``trimesh`` has no public hook for this extension as of 4.8.x.
    """
    with open(path, "rb") as f:
        blob = f.read()
    magic, version, _total = struct.unpack("<4sII", blob[:12])
    if magic != b"glTF":
        raise ValueError(f"{path}: not a GLB (magic={magic!r})")
    chunk0_len = struct.unpack("<I", blob[12:16])[0]
    chunk0_type = blob[16:20]
    if chunk0_type != b"JSON":
        raise ValueError(f"{path}: first chunk is not JSON ({chunk0_type!r})")
    json_bytes = blob[20 : 20 + chunk0_len]
    rest = blob[20 + chunk0_len :]
    gltf = json.loads(json_bytes.decode("utf-8"))

    touched = False
    for mat in gltf.get("materials", []):
        if mat.get("name") == material_name:
            ext = mat.setdefault("extensions", {})
            if "KHR_materials_unlit" not in ext:
                ext["KHR_materials_unlit"] = {}
            touched = True
    if not touched:
        return  # nothing to do; leave the file untouched

    used = gltf.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in used:
        used.append("KHR_materials_unlit")

    new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # JSON chunk MUST be 4-byte aligned; spec requires space (0x20) padding.
    pad = (4 - (len(new_json) % 4)) % 4
    new_json += b" " * pad
    new_total = 12 + 8 + len(new_json) + len(rest)

    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", version, new_total))
        f.write(struct.pack("<I", len(new_json)))
        f.write(b"JSON")
        f.write(new_json)
        f.write(rest)
