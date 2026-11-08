# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import numpy as np
from torch import Tensor


def view_matrix_from_string(convention: str) -> np.ndarray:
    """Return a 3×3 matrix mapping coordinates expressed in *convention* to the
    default RFU (Right-Forward-Up) convention used by our visualization libs.

    The *convention* string must have three characters.  Each denotes the
    positive direction of the x-, y- and z-axis respectively:

        R / L – Right (+X)  /  Left (−X)
        F / B – Forward (+Y) /  Back (−Y)
        U / D – Up (+Z)     /  Down (−Z)

    Example
    -------
    >>> view_matrix_from_string("RDF")
    array([[ 1,  0,  0],
           [ 0,  0,  1],
           [ 0, -1,  0]])

    The matrix *M* satisfies::

        coords_rfu = M @ coords_in_convention

    so when *convention* == "RFU" we simply return the identity.
    """

    if len(convention) != 3:
        raise ValueError("Coordinate convention string must contain exactly 3 characters, e.g. 'RFU'.")

    mapping = {
        "R": np.array([1, 0, 0]),  # +X
        "L": np.array([-1, 0, 0]),  # -X
        "F": np.array([0, 1, 0]),  # +Y
        "B": np.array([0, -1, 0]),  # -Y
        "U": np.array([0, 0, 1]),  # +Z
        "D": np.array([0, 0, -1]),  # -Z
    }

    cols = []
    for c in convention.upper():
        if c not in mapping:
            raise ValueError(f"Invalid axis specifier '{c}' in coordinate convention '{convention}'.")
        cols.append(mapping[c])
    return np.stack(cols, axis=1).astype(float)  # shape (3,3)


def calculate_auto_image_plane_distance(
    cameras,
    points: Optional[Union[Tensor, np.ndarray]] = None,
    radius_scale_factor: float = 0.05,
) -> float:
    """Calculate automatic image plane distance based on scene extent.

    Uses point cloud data when available, otherwise falls back to camera positions.
    For dense point clouds, the radius uses a **median** center and **99.5th
    percentile** distance so a few outlier points (bad depths) do not blow up
    Rerun's camera frustum / image-plane distance; sparse camera-only paths use
    mean + max as before.

    Args:
        cameras: Camera objects containing poses
        points: Optional 3D point positions
        radius_scale_factor: Scale factor to apply to scene radius

    Returns:
        Calculated image plane distance as percentage of scene radius
    """
    if points is not None:
        positions = points.cpu().numpy() if isinstance(points, Tensor) else points
    elif isinstance(cameras, dict):
        all_positions = []
        for _, cams in cameras.items():
            for camera in cams:
                all_positions.append(camera.camera_to_worlds[:3, 3].cpu().numpy())
        positions = np.array(all_positions)
    else:
        positions = np.array([camera.camera_to_worlds[:3, 3].cpu().numpy() for camera in cameras])

    assert len(positions) > 0, "No positions to calculate scene radius"

    # Dense point clouds: mean + max is dominated by a handful of bad depths (Rerun
    # image planes / framing use this scalar). Camera-only fallbacks stay mean + max.
    from_points = points is not None
    n = len(positions)
    if from_points and n >= 8:
        centroid = np.median(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        scene_radius = float(np.percentile(distances, 99.5))
    else:
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        scene_radius = float(np.max(distances))
    return scene_radius * radius_scale_factor
