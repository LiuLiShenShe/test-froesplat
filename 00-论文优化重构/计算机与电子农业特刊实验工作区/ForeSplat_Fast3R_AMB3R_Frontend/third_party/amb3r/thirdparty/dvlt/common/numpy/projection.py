# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy pinhole-camera projection helpers.

These are tiny pinhole-projection / unprojection routines, separated from
their torch counterparts in :mod:`dvlt.common.projection` because they're
used inside dataset parsers that operate on raw numpy arrays before any
tensor conversion happens.
"""

import numpy as np


def depth_to_cam_coords_points(
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Unproject a depth map to camera-frame XYZ points.

    Assumes a standard pinhole camera model with zero skew. Each pixel
    ``(u, v)`` is unprojected to ``((u - cx) * z / fx, (v - cy) * z / fy, z)``
    where ``z`` is the pixel's depth value.

    Args:
        depth_map: ``(H, W)`` z-depth values.
        intrinsics: ``(3, 3)`` pinhole intrinsics matrix
            ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.

    Returns:
        ``(H, W, 3)`` float32 array of camera-frame XYZ coordinates.
    """
    assert intrinsics.shape == (3, 3), "intrinsics must be 3x3"
    assert intrinsics[0, 1] == 0 and intrinsics[1, 0] == 0, "intrinsics must be skew-free"

    H, W = depth_map.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    vs, us = np.mgrid[0:H, 0:W].astype(np.float32)
    z = depth_map.astype(np.float32, copy=False)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def project_points(
    points: np.ndarray,
    intrinsics: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Project 3D points to 2D pixel coordinates with pinhole intrinsics.

    Args:
        points: ``(N, 3)`` or ``(B, N, 3)`` 3D coordinates.
        intrinsics: ``(3, 3)`` or ``(B, 3, 3)`` intrinsic camera matrices.
        eps: Floor for the depth coordinate before division (avoids /0).

    Returns:
        ``(N, 2)`` or ``(B, N, 2)`` 2D pixel coordinates.
    """
    assert points.shape[-1] == 3, "input must have last dim 3"

    z_safe = np.where(points[..., 2] == 0, eps, points[..., 2])
    hom = points / z_safe[..., None]

    if hom.ndim == 2:
        assert intrinsics.ndim == 2, "got batched intrinsics for un-batched points"
        K_t = intrinsics.T
    elif hom.ndim == 3:
        K = intrinsics if intrinsics.ndim == 3 else np.expand_dims(intrinsics, 0)
        K_t = np.transpose(K, (0, 2, 1))
    else:
        raise ValueError(f"points must be (N, 3) or (B, N, 3), got {points.shape}")

    return (hom @ K_t)[..., :2]
