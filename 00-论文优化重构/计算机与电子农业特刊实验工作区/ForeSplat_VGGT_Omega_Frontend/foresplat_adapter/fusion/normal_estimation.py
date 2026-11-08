"""Per-view normal estimation from VGGT-Omega point maps.

Normals are estimated **from the local neighbourhood of each view's own point
map** (no global kNN, no cross-view smoothing), then oriented towards that
view's camera centre.  Alongside the normal we report

* a *surface quality* in [0, 1] (planarity of the local covariance: 1 = perfect
  plane, 0 = volumetric/noisy neighbourhood), and
* an *edge flag* marking pixels that sit on a depth discontinuity.

The edge flag is what makes the later fusion edge preserving: edge pixels are
fused with a finer voxel size and a higher weight so leaf silhouettes keep
their sampling density instead of being averaged away.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_view_normals", "depth_edge_mask"]


def _shifted(array: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """Shift ``array`` by (dy, dx) with ``fill`` outside the image."""
    h, w = array.shape[:2]
    out = np.full_like(array, fill)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    out[src_y0 + dy : src_y1 + dy, src_x0 + dx : src_x1 + dx] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def depth_edge_mask(depth: np.ndarray, valid: np.ndarray, rtol: float = 0.03) -> np.ndarray:
    """Pixels whose depth differs from a 4-neighbour by more than ``rtol`` (relative)."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    safe = np.where(valid, depth, np.nan)
    edge = np.zeros(depth.shape, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = _shifted(safe, dy, dx, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(neighbour - safe) / np.maximum(np.abs(safe), 1e-6)
        jump = np.isfinite(rel) & (rel > rtol)
        edge |= jump
    return edge & valid


def estimate_view_normals(
    points: np.ndarray,
    valid: np.ndarray,
    camera_center: np.ndarray,
    depth: np.ndarray | None = None,
    radius: int = 2,
    edge_rtol: float = 0.03,
    min_neighbours: int = 3,
) -> dict:
    """Estimate world-space normals for the valid pixels of one view.

    Parameters
    ----------
    points : (H, W, 3) float
        World-space point map of the view (depth back-projection).
    valid : (H, W) bool
        Pixels that carry usable geometry (foreground & finite & valid depth).
    camera_center : (3,) float
        Camera centre of the view in the same world frame.
    depth : (H, W) float, optional
        Depth map used for the edge flag (falls back to the point-map depth
        along the view direction when omitted).

    Returns
    -------
    dict with ``normals`` (H, W, 3) float32, ``quality`` (H, W) float32,
    ``edge`` (H, W) bool, ``neighbour_count`` (H, W) int16.  Invalid pixels have
    zero normals / quality and are flagged by ``valid``.
    """
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    h, w, _ = points.shape
    radius = max(1, int(radius))

    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if not (dy == 0 and dx == 0)
    ]

    # Accumulate the first and second moments of the local neighbourhood.
    count = valid.astype(np.float32)
    sum_p = np.where(valid[..., None], points, 0.0).astype(np.float32)
    sum_outer = np.zeros((h, w, 6), dtype=np.float32)
    p = points
    m = valid
    for k, (i, j) in enumerate(_UPPER_TRIPLES):
        sum_outer[..., k] = np.where(m, p[..., i] * p[..., j], 0.0)

    for dy, dx in offsets:
        shifted_p = _shifted(p, dy, dx, 0.0)
        shifted_m = _shifted(m, dy, dx, False)
        count += shifted_m
        sum_p += np.where(shifted_m[..., None], shifted_p, 0.0)
        for k, (i, j) in enumerate(_UPPER_TRIPLES):
            sum_outer[..., k] += np.where(shifted_m, shifted_p[..., i] * shifted_p[..., j], 0.0)

    safe_count = np.maximum(count, 1.0)
    mean = sum_p / safe_count[..., None]
    cov = np.zeros((h, w, 3, 3), dtype=np.float32)
    for k, (i, j) in enumerate(_UPPER_TRIPLES):
        value = sum_outer[..., k] / safe_count - mean[..., i] * mean[..., j]
        cov[..., i, j] = value
        cov[..., j, i] = value

    # Eigen-decomposition of every 3x3 covariance (smallest eigenvector = normal).
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    normals = eigvecs[..., 0].astype(np.float32)

    # Orient towards the camera (leaves seen from their own side face the camera).
    direction = np.asarray(camera_center, dtype=np.float32)[None, None, :] - points
    flip = np.sum(normals * direction, axis=-1, keepdims=True) < 0
    normals = np.where(flip, -normals, normals)

    # Planarity: (l2 - l3) / (l1 + eps); 1 = flat surface.
    l1 = np.maximum(eigvals[..., 2], 1e-12)
    planarity = (eigvals[..., 1] - eigvals[..., 0]) / l1
    quality = np.clip(planarity, 0.0, 1.0).astype(np.float32)

    enough = (count >= min_neighbours) & valid
    normals = np.where(enough[..., None], normals, 0.0).astype(np.float32)
    quality = np.where(enough, quality, 0.0).astype(np.float32)

    if depth is None:
        # Fall back: depth = distance along the camera ray.
        ray = points - camera_center[None, None, :]
        depth = np.linalg.norm(ray, axis=-1)
    edge = depth_edge_mask(depth, valid, rtol=edge_rtol)

    return {
        "normals": normals,
        "quality": quality,
        "edge": edge,
        "neighbour_count": np.minimum(count, 32767).astype(np.int16),
    }


_UPPER_TRIPLES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))