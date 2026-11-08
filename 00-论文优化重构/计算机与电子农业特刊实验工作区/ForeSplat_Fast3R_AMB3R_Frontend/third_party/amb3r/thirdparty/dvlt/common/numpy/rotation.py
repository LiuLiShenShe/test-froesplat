# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def so3_relative_angle(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Compute the angle between 2 rotation matrices.
    Args:
        R1: the 1st rotation matrix with shape (B, 3, 3)
        R2: the 2nd rotation matrix with shape (B, 3, 3)
    Returns:
        The rotation matrix angle in radians with shape (B,)
    """
    # (Trace(R1.T @ R2) - 1)/2
    prod = np.matmul(np.swapaxes(R1, -2, -1), R2)
    cos = (np.trace(prod, axis1=-2, axis2=-1) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    diff = np.abs(np.arccos(cos))
    return diff


def quat_to_mat(quaternions: np.ndarray) -> np.ndarray:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as array of shape (..., 4).

    Returns:
        Rotation matrices as array of shape (..., 3, 3).
    """
    i, j, k, r = np.split(quaternions, 4, axis=-1)
    i, j, k, r = i.squeeze(-1), j.squeeze(-1), k.squeeze(-1), r.squeeze(-1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = np.moveaxis(matrix.reshape((*batch_dim, 9)), -1, 0)

    q_abs = _sqrt_positive_part(
        np.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = 0.1
    quat_candidates = quat_by_rijk / (2.0 * np.maximum(q_abs[..., None], flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(axis=-1)
    # Create one-hot mask for advanced indexing
    mask = np.eye(4)[indices]
    out = (quat_candidates * mask[..., None]).sum(axis=-2)

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: np.ndarray) -> np.ndarray:
    """
    Returns np.sqrt(np.maximum(0, x))
    but handles negative values gracefully.
    """
    return np.sqrt(np.maximum(0, x))


def standardize_quaternion(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as array of shape (..., 4).

    Returns:
        Standardized quaternions as array of shape (..., 4).
    """
    return np.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)
