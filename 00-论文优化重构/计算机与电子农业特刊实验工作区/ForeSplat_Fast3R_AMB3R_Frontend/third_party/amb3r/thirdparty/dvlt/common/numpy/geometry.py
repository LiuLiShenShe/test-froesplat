# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy depth / point-cloud helpers used by dataset parsers.

These mirror the torch implementations in :mod:`dvlt.common.geometry`, but
work on raw numpy arrays. They are invoked by the dataset loading path,
before tensor conversion.
"""

from typing import Optional, Tuple

import numpy as np

from dvlt.common.numpy.projection import depth_to_cam_coords_points


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsics_c2w: np.ndarray,
    intrinsics: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject a depth map to world-frame XYZ points and a validity mask.

    The pipeline is: pinhole unprojection to camera frame, followed by a
    rigid transform with the supplied camera-to-world extrinsics. The
    validity mask flags pixels whose depth strictly exceeds ``eps``.

    Args:
        depth_map: ``(H, W)`` depth in metric units.
        extrinsics_c2w: ``(3, 4)`` or ``(4, 4)`` camera-to-world transform.
        intrinsics: ``(3, 3)`` pinhole intrinsics.
        eps: Threshold above which a pixel's depth counts as valid.

    Returns:
        Tuple ``(world_xyz, cam_xyz, mask)`` with shapes
        ``(H, W, 3)``, ``(H, W, 3)``, ``(H, W)``.
    """
    valid = depth_map > eps

    cam_xyz = depth_to_cam_coords_points(depth_map, intrinsics)

    R_c2w = extrinsics_c2w[:3, :3]
    t_c2w = extrinsics_c2w[:3, 3]
    world_xyz = cam_xyz @ R_c2w.T + t_c2w

    return world_xyz, cam_xyz, valid


def transform_points(
    transformation: np.ndarray,
    points: np.ndarray,
    dim: Optional[int] = None,
) -> np.ndarray:
    """Apply a 4x4 rigid (or affine) transformation to a set of points.

    Args:
        transformation: ``(..., 4, 4)`` transform.
        points: ``(..., 3)`` points (broadcasts against the batch dims of
            ``transformation``).
        dim: Number of leading point dims to return (default ``points.shape[-1]``).

    Returns:
        Transformed points with shape ``(..., dim)``.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    return points[..., :dim].reshape(*initial_shape, dim)
