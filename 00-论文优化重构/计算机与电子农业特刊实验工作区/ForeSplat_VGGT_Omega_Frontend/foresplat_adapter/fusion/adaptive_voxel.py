"""Adaptive voxel-size selection for multi-view fusion.

The voxel size is derived from the scene itself (never hand-tuned per plant):

1. A robust nearest-neighbour scale ``d_med`` is measured on a random
   subsample of the raw candidate points (median 1-NN distance).
2. A geometric seed ``v_seed = scene_extent / cbrt(target_points)`` captures
   the density needed to reach the requested target point count.
3. The final voxel is ``clip(max(v_seed, alpha * d_med), v_min, v_max)`` so
   that voxels never become smaller than the sensor's local point spacing
   (which would create duplicate points on smooth surfaces).
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_nn_scale", "choose_voxel_size", "voxel_keys", "count_unique_voxels"]


def _random_subsample(points: np.ndarray, max_samples: int, seed: int = 0) -> np.ndarray:
    if points.shape[0] <= max_samples:
        return points
    rng = np.random.default_rng(seed)
    pick = rng.choice(points.shape[0], size=max_samples, replace=False)
    return points[pick]


def estimate_nn_scale(points: np.ndarray, max_samples: int = 100_000, seed: int = 0) -> float:
    """Median 1-NN distance of a random subsample of ``points`` (KD-tree)."""
    pts = _random_subsample(points, max_samples, seed=seed)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        dist, _ = tree.query(pts, k=2, workers=-1)
        return float(np.median(dist[:, 1]))
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        pass
    # Chunked brute force fallback (bounded memory).
    chunk = max(1, int(4e6 // n))
    best = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        diff = pts[start:stop, None, :] - pts[None, :, :]
        dist = np.sqrt((diff * diff).sum(-1))
        dist[np.arange(start, stop) - start, np.arange(start, stop)] = np.inf
        best[start:stop] = dist.min(axis=1)
    return float(np.median(best))


def count_unique_voxels(points: np.ndarray, voxel_size: float) -> int:
    """Number of distinct voxels covered by ``points`` at ``voxel_size``."""
    keys = voxel_keys(points, voxel_size)
    return int(np.unique(keys).size)


def choose_voxel_size(
    points: np.ndarray,
    target_points: int,
    nn_scale: float | None = None,
    alpha: float = 1.0,
    min_voxel: float = 1e-4,
    max_voxel: float | None = None,
    seed: int = 0,
) -> dict:
    """Pick a voxel size so the fused cloud lands near ``target_points``.

    Returns a dict with the voxel size, the measured nn scale, the geometric
    seed, and the expected fused count.
    """
    extent = float(np.ptp(points, axis=0).max())
    if nn_scale is None:
        nn_scale = estimate_nn_scale(points, seed=seed)
    seed_voxel = extent / max(target_points, 1) ** (1.0 / 3.0)
    voxel = max(seed_voxel, alpha * nn_scale)
    if max_voxel is not None:
        voxel = min(voxel, max_voxel)
    voxel = max(voxel, min_voxel)
    # Refine with actual voxel counts (binary search within a factor of 2).
    # ``count`` decreases as the voxel grows: too many points -> grow the voxel.
    expected = count_unique_voxels(points, voxel)
    low, high = voxel / 3.0, voxel * 3.0
    for _ in range(12):
        if abs(expected - target_points) <= 0.05 * target_points:
            break
        if expected > target_points:
            low = voxel
        else:
            high = voxel
        voxel = float(np.sqrt(low * high))
        expected = count_unique_voxels(points, voxel)
    return {
        "voxel_size": float(voxel),
        "nn_scale": float(nn_scale),
        "geometric_seed_voxel": float(seed_voxel),
        "scene_extent": extent,
        "expected_fused_points": int(expected),
    }


def voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deterministic int64 hash keys of the voxel each point falls into."""
    scaled = np.floor(points / float(voxel_size)).astype(np.int64)
    # Pack three indices into a single int64 key.  Indices are clamped so the
    # packing stays injective and cannot overflow for absurd coordinates.
    limit = (1 << 20) - 1
    scaled = np.clip(scaled, -limit, limit) + limit
    ix, iy, iz = scaled[:, 0], scaled[:, 1], scaled[:, 2]
    base = 2 * limit + 1
    return (ix * base + iy) * base + iz