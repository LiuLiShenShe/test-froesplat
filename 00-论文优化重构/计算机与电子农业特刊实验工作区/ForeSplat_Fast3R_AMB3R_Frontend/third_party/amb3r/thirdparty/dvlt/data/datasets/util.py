# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numpy-side image / depth / camera-matrix utilities used by the training and
evaluation dataset loaders.

The helpers in this file all operate on raw numpy arrays (image, depth, 3x3
intrinsics, 3x4 / 4x4 camera-to-world extrinsics) and are intentionally kept
small and composable so that individual dataset parsers can mix-and-match the
pieces they need (e.g. some parsers crop-around-principal-point first, others
go straight to the resize step).

This file is dvlt-authored. The set of operations covered here — principal-
point cropping, resize-with-overshoot, 90° image rotations and the matching
intrinsics / extrinsics fixups — is broadly standard in multi-view dataset
pipelines. Where named arguments, default values, or rounding conventions
matter, dvlt follows whichever behaviour empirically worked best on the
train/eval datasets we ship.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from dvlt.common.numpy.rotation import so3_relative_angle
from dvlt.common.pose import inverse_pose


# Pillow resampling constants. Kept module-level so callers can pass them
# through directly when they need to do their own resize step.
_LANCZOS = PIL.Image.Resampling.LANCZOS
_BICUBIC = PIL.Image.Resampling.BICUBIC


# -----------------------------------------------------------------------------
# Random-augmentation factor sampler
# -----------------------------------------------------------------------------


def sample_scale_aug_factor(scale_aug_max: float = 0.3) -> float:
    """Sample a zoom-in scale factor from ``Triangular(0, 0, scale_aug_max)``.

    The triangular distribution with both ``left`` and ``mode`` set to 0 is
    monotonically decreasing on ``[0, scale_aug_max]``, which biases samples
    toward "no augmentation" while still occasionally producing a larger
    crop. The intended downstream use is :func:`resize_with_overshoot` via
    its ``scale_aug_factor`` argument.
    """
    return float(np.random.triangular(0.0, 0.0, scale_aug_max))


# -----------------------------------------------------------------------------
# Finite-value sanity check
# -----------------------------------------------------------------------------


def check_all_finite(sample: Dict[str, Any], sample_name: Optional[str] = None) -> bool:
    """Return ``True`` iff every torch.Tensor in ``sample`` is finite.

    Prints a single diagnostic line on the first non-finite tensor found and
    returns ``False`` so the caller can short-circuit retries.
    """
    label = sample_name if sample_name is not None else "sample"
    for key, value in sample.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            print(f"Non-finite value found in {key} of {label}")
            return False
    return True


# -----------------------------------------------------------------------------
# Color / appearance augmentation pipeline (dvlt-authored).
#
# Thin wrapper around torchvision's ColorJitter / RandomGrayscale /
# RandomApply(GaussianBlur). The kwargs are simple enable-toggles for each
# step; the assembly is a filter-out-None list comprehension over
# independently constructed steps.
# -----------------------------------------------------------------------------

# Defaults are intentionally mild compared to upstream ColorJitter recipes —
# more saturation/contrast tends to hurt photometric depth metrics on indoor
# scenes (ScanNet++, ETH3D).
_DEFAULT_JITTER_PARAMS: Dict[str, float] = {
    "brightness": 0.3,
    "contrast": 0.4,
    "saturation": 0.2,
    "hue": 0.1,
    "p": 0.9,
}

# Independent per-frame probability for the grayscale / blur side-channels.
# Both deliberately small so they only fire on ~5% of frames.
_GRAYSCALE_PROB: float = 0.05
_BLUR_PROB: float = 0.05
_BLUR_KERNEL_SIZE: int = 5
_BLUR_SIGMA_RANGE: Tuple[float, float] = (0.1, 1.0)


def _make_color_jitter_step(jitter_params: Optional[Dict[str, float]]) -> Optional[transforms.RandomApply]:
    """Return a RandomApply([ColorJitter(...)], p=p) or None when ``p == 0``."""
    cfg = dict(_DEFAULT_JITTER_PARAMS)
    if isinstance(jitter_params, dict):
        cfg.update(jitter_params)
    if cfg["p"] <= 0.0:
        return None
    jitter = transforms.ColorJitter(
        brightness=cfg["brightness"],
        contrast=cfg["contrast"],
        saturation=cfg["saturation"],
        hue=cfg["hue"],
    )
    return transforms.RandomApply([jitter], p=cfg["p"])


def build_appearance_aug(
    jitter_params: Optional[Dict[str, float]] = None,
    enable_grayscale: bool = True,
    enable_gaussian_blur: bool = False,
) -> Optional[transforms.Compose]:
    """Assemble dvlt's per-frame appearance augmentation pipeline.

    Operates on ``[C, H, W]`` float tensors in ``[0, 1]``. Returns ``None``
    when no step is enabled, so callers can short-circuit.

    Args:
        jitter_params: Optional overrides for the ColorJitter knobs
            ``{"brightness", "contrast", "saturation", "hue", "p"}``. Missing
            keys fall back to dvlt's mild defaults. Setting ``p`` to ``0``
            drops the jitter step.
        enable_grayscale: Append a low-probability random grayscale step.
        enable_gaussian_blur: Append a low-probability random Gaussian-blur step.
    """
    grayscale_step = transforms.RandomGrayscale(p=_GRAYSCALE_PROB) if enable_grayscale else None

    if enable_gaussian_blur:
        blur_step = transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=_BLUR_KERNEL_SIZE, sigma=_BLUR_SIGMA_RANGE)],
            p=_BLUR_PROB,
        )
    else:
        blur_step = None

    candidate_steps = [
        _make_color_jitter_step(jitter_params),
        grayscale_step,
        blur_step,
    ]
    steps = [s for s in candidate_steps if s is not None]
    return transforms.Compose(steps) if steps else None


# -----------------------------------------------------------------------------
# Scene normalization (translate so first camera is the origin, optionally
# rescale so that the average point distance is ~1)
# -----------------------------------------------------------------------------


def compute_normalization_transform(
    extrinsics_c2w: Tensor,
    world_points: Optional[Tensor] = None,
    point_masks: Optional[Tensor] = None,
    scale_by_points: bool = True,
) -> Tuple[Tensor, float]:
    """Translate the scene into camera-0's frame and pick a global scale.

    The transform applied to world-space arrays is::

        world' = (world @ R0.T + t0) / s

    where ``[R0 | t0] = inverse_pose(extrinsics_c2w[0])`` is the world-to-cam0
    transform and ``s`` is a positive scalar derived either from valid
    world-point distances (if available) or from the mean relative camera
    translation magnitude.

    Args:
        extrinsics_c2w: ``(S, 4, 4)`` camera-to-world poses for ``S`` frames.
        world_points: optional ``(S, H, W, 3)`` per-pixel world coordinates.
        point_masks: optional ``(S, H, W)`` validity mask aligned with
            ``world_points``.
        scale_by_points: when False (or when no valid masked points are
            present *and* there's only a single frame) ``s`` defaults to 1.

    Returns:
        ``(world2cam0, scale)``. ``world2cam0`` is ``(4, 4)`` and is meant to be
        left-multiplied onto pose tensors *before* dividing translations by
        ``scale``. ``scale`` is a python float clamped to ``[1e-3, 1e3]``.
    """
    world2cam0 = inverse_pose(extrinsics_c2w[0])

    scale: float = 1.0
    if scale_by_points and world_points is not None and point_masks is not None:
        # Bring world points into the cam0 frame to measure scale.
        R0 = world2cam0[:3, :3]
        t0 = world2cam0[:3, 3]
        pts_cam0 = torch.einsum("shwi,ij->shwj", world_points, R0.T) + t0.view(1, 1, 1, 3)

        if int(point_masks.sum().item()) > 0:
            avg = pts_cam0[point_masks].norm(dim=-1).mean()
        else:
            # No valid 3D points: fall back to relative camera translations.
            other_poses = world2cam0 @ extrinsics_c2w[1:]
            if other_poses.shape[0] == 0:
                avg = torch.ones((), device=extrinsics_c2w.device)
            else:
                avg = other_poses[:, :3, 3].norm(dim=-1).mean()

        scale = float(avg.clamp(min=1e-3, max=1e3).item())

    return world2cam0, scale


# -----------------------------------------------------------------------------
# Principal-point centered crop
# -----------------------------------------------------------------------------


def crop_around_principal_point(
    image: np.ndarray,
    depth_map: Optional[np.ndarray],
    intrinsic: np.ndarray,
    target_hw,
    filepath: Optional[str] = None,
    strict: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Crop ``image``/``depth_map`` centered on the camera's principal point.

    The crop window is centered on ``(cy, cx) = (intrinsic[1,2], intrinsic[0,2])``
    in *image* coordinates (note: numpy index order is row-major). When
    ``strict=False``, the window may be tighter than ``target_hw`` (e.g. when
    the principal point is close to an image edge), and the function returns
    the largest principal-point-centered square that fits inside the image.
    When ``strict=True``, the output is zero-padded as needed to hit
    ``target_hw`` exactly.

    The intrinsics matrix is shifted to keep the optical center consistent
    with the cropped pixel grid. Depth maps follow the image crop.

    Args:
        image: input image array of shape ``(H, W, C)`` or ``(H, W)``.
        depth_map: matching depth array of shape ``(H, W)`` (or ``None``).
        intrinsic: ``(3, 3)`` pinhole intrinsics, modified out-of-place.
        target_hw: ``(H_target, W_target)``.
        filepath: only used in strict-mode diagnostic prints.
        strict: when True, zero-pad to exactly ``target_hw``.

    Returns:
        ``(cropped_image, cropped_depth_map, cropped_intrinsic)``.
    """
    src_h, src_w = image.shape[:2]
    tgt_h, tgt_w = int(target_hw[0]), int(target_hw[1])
    K = intrinsic.copy()

    if src_h < tgt_h or src_w < tgt_w:
        msg = f"Source image too small for principal-point crop: " f"src=({src_h}, {src_w}), target=({tgt_h}, {tgt_w})"
        print(msg)
        raise AssertionError(msg)

    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # How far we can extend from the principal point in each direction.
    if strict:
        half_w = min(tgt_w / 2.0, cx)
        half_h = min(tgt_h / 2.0, cy)
    else:
        half_w = min(tgt_w / 2.0, cx, src_w - cx)
        half_h = min(tgt_h / 2.0, cy, src_h - cy)

    half_w_i = int(np.floor(half_w))
    half_h_i = int(np.floor(half_h))
    cx_i = int(np.floor(cx))
    cy_i = int(np.floor(cy))

    x0 = cx_i - half_w_i
    y0 = cy_i - half_h_i
    assert x0 >= 0 and y0 >= 0, f"Negative crop offset (x0={x0}, y0={y0})"

    if strict:
        x1 = x0 + tgt_w
        y1 = y0 + tgt_h
    else:
        x1 = x0 + 2 * half_w_i
        y1 = y0 + 2 * half_h_i

    image_cropped = image[y0:y1, x0:x1, ...] if image.ndim == 3 else image[y0:y1, x0:x1]
    depth_cropped = depth_map[y0:y1, x0:x1] if depth_map is not None else None

    # Shift the principal point so it remains pinned to the same pixel.
    K[0, 2] = cx - x0
    K[1, 2] = cy - y0

    if strict and tuple(image_cropped.shape[:2]) != (tgt_h, tgt_w):
        print(f"{filepath} does not meet the target shape")
        cur_h, cur_w = image_cropped.shape[:2]
        pad_h = tgt_h - cur_h
        pad_w = tgt_w - cur_w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Cropped image larger than target: " f"cropped=({cur_h}, {cur_w}), target=({tgt_h}, {tgt_w})."
            )
        pad_spec_img = ((0, pad_h), (0, pad_w), (0, 0)) if image_cropped.ndim == 3 else ((0, pad_h), (0, pad_w))
        image_cropped = np.pad(image_cropped, pad_width=pad_spec_img, mode="constant", constant_values=0)
        if depth_cropped is not None:
            depth_cropped = np.pad(
                depth_cropped,
                pad_width=((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

    return image_cropped, depth_cropped, K


# -----------------------------------------------------------------------------
# Resize with overshoot (output is at least target_hw + safe_bound on the
# tighter axis, then a strict principal-point crop tightens to target_hw)
# -----------------------------------------------------------------------------


def resize_with_overshoot(
    image: np.ndarray,
    depth_map: Optional[np.ndarray],
    intrinsic: np.ndarray,
    target_hw,
    original_hw,
    pixel_center: bool = True,
    safe_bound: int = 4,
    scale_aug_factor: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Resize ``image``/``depth_map`` to slightly larger than ``target_hw``.

    The function computes a single uniform scale factor ``s = max((target_hw + safe_bound) / original_hw)``
    so that *both* output dimensions are at least ``target_hw + safe_bound``.
    Depth is resized with nearest-neighbor; RGB with LANCZOS (downscale) or
    BICUBIC (upscale). Intrinsics are scaled by the *actual* resize ratio
    (which can differ slightly from ``s`` due to floor() rounding of the
    output resolution).

    Args:
        image: ``(H, W, C)``.
        depth_map: ``(H, W)`` or None.
        intrinsic: ``(3, 3)``.
        target_hw: target ``(H, W)``.
        original_hw: source ``(H, W)`` (passed explicitly because callers
            sometimes resize before calling, e.g. when chaining with crops).
        pixel_center: when True, applies a ``+0.5 / -0.5`` shift around the
            principal point so the scaling acts on pixel centers rather than
            corners.
        safe_bound: extra integer margin added to ``target_hw`` before computing
            the scale factor.
        scale_aug_factor: when not ``None``, augments ``safe_bound`` by
            ``scale_aug_factor * max(target_hw)``; this implements the zoom-in
            augmentation. See :func:`sample_scale_aug_factor`.

    Returns:
        ``(resized_image, resized_depth_map, resized_intrinsic)``.
    """
    target_hw_arr = np.asarray(target_hw, dtype=np.float64)
    orig_hw_arr = np.asarray(original_hw, dtype=np.float64)

    bound = float(safe_bound)
    if scale_aug_factor is not None:
        bound += float(scale_aug_factor) * float(target_hw_arr.max())

    scale = float(np.max((target_hw_arr + bound) / orig_hw_arr))
    K = intrinsic.copy()

    # Pillow resize uses (W, H) ordering. We snap the new size to floor(scale * src).
    pil_img = Image.fromarray(image)
    src_w, src_h = pil_img.size
    out_w = int(np.floor(src_w * scale))
    out_h = int(np.floor(src_h * scale))

    resample = _LANCZOS if scale < 1.0 else _BICUBIC
    pil_img = pil_img.resize((out_w, out_h), resample=resample)
    image_out = np.array(pil_img)

    depth_out = None
    if depth_map is not None:
        depth_out = cv2.resize(
            depth_map,
            (out_w, out_h),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

    # Use the realized output shape rather than the requested scale; this
    # avoids drift when floor() rounded the output resolution down.
    actual_h, actual_w = image_out.shape[:2]
    actual_scale = float(max(actual_h / orig_hw_arr[0], actual_w / orig_hw_arr[1]))

    if pixel_center:
        # Shift to pixel-centered coords, scale, then shift back.
        K[0, 2] += 0.5
        K[1, 2] += 0.5
    K[:2, :] *= actual_scale
    if pixel_center:
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5

    if depth_out is not None:
        assert (
            image_out.shape[:2] == depth_out.shape[:2]
        ), f"Resize produced mismatched shapes: image={image_out.shape[:2]}, depth={depth_out.shape[:2]}"

    return image_out, depth_out, K


# -----------------------------------------------------------------------------
# 90-degree image rotation and matching camera-matrix fixups
# -----------------------------------------------------------------------------

# In-image-plane rotation of ±90° expressed as a 3x3 rigid rotation. The
# choice of sign matches the convention "clockwise" = rotate the *image* CW
# when viewed by an observer in front of the camera.
_R90_CW = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
_R90_CCW = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)


def rotate_array_90(arr: np.ndarray, clockwise: bool) -> np.ndarray:
    """Rotate ``arr`` 90 degrees in-place-of-an-image-grid using transpose+flip.

    Accepts 2D ``(H, W)`` or 3D ``(H, W, C)`` arrays. Returns a contiguous copy.
    """
    if arr.ndim == 2:
        axes = (1, 0)
    elif arr.ndim == 3:
        axes = (1, 0, 2)
    else:
        raise ValueError(f"rotate_array_90 expects 2D or 3D arrays, got {arr.ndim}D")

    rotated = np.transpose(arr, axes)
    flip_axis = 1 if clockwise else 0
    return np.ascontiguousarray(np.flip(rotated, axis=flip_axis))


def rotate_frame_90(
    image: np.ndarray,
    depth_map: Optional[np.ndarray],
    extrinsics_c2w: np.ndarray,
    intrinsics: np.ndarray,
    clockwise: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Apply a 90° in-plane rotation to ``image``, ``depth_map``, and camera matrices.

    Args:
        image: ``(H, W, C)``.
        depth_map: ``(H, W)`` or None.
        extrinsics_c2w: ``(3, 4)`` or ``(4, 4)`` camera-to-world pose.
        intrinsics: ``(3, 3)``.
        clockwise: which direction to rotate the image.

    Returns:
        Same set of tensors after rotation, with intrinsics and extrinsics
        adjusted so that projecting world points through the new ``(K', E')``
        matches the rotated image.
    """
    src_h, src_w = image.shape[:2]
    rotated_image = rotate_array_90(image, clockwise)
    rotated_depth = rotate_array_90(depth_map, clockwise) if depth_map is not None else None

    new_K = rotate_intrinsics_90(intrinsics, image_width=src_w, image_height=src_h, clockwise=clockwise)
    new_E = rotate_extrinsics_90(extrinsics_c2w, clockwise=clockwise)
    return rotated_image, rotated_depth, new_E, new_K


def rotate_extrinsics_90(extrinsics_c2w: np.ndarray, clockwise: bool) -> np.ndarray:
    """Update a camera-to-world pose for a 90° in-plane image rotation.

    The image rotation acts on the *camera frame*; in world-to-camera form the
    new pose is ``R_rot @ E_w2c``. We invert in, transform, invert out.
    """
    # Promote to 4x4 so that the round-trip preserves the homogeneous row.
    E_c2w = extrinsics_c2w
    if E_c2w.shape[0] == 3:
        bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=E_c2w.dtype)
        E_c2w = np.vstack((E_c2w, bottom))

    E_w2c = inverse_pose(E_c2w)
    R_img = (_R90_CW if clockwise else _R90_CCW).astype(E_c2w.dtype)

    new_E_w2c = np.eye(4, dtype=E_c2w.dtype)
    new_E_w2c[:3, :3] = R_img @ E_w2c[:3, :3]
    new_E_w2c[:3, 3] = R_img @ E_w2c[:3, 3]

    return inverse_pose(new_E_w2c).astype(extrinsics_c2w.dtype)


def rotate_intrinsics_90(
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    clockwise: bool,
) -> np.ndarray:
    """Update a 3x3 intrinsics matrix for a 90° in-plane image rotation.

    A 90° rotation swaps the focal-length axes (fx <-> fy) and remaps the
    principal point inside the new (rotated) image grid.
    """
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    K_new = np.eye(3, dtype=intrinsics.dtype)
    # Focals always swap regardless of direction.
    K_new[0, 0] = fy
    K_new[1, 1] = fx
    if clockwise:
        K_new[0, 2] = image_height - cy
        K_new[1, 2] = cx
    else:
        K_new[0, 2] = cy
        K_new[1, 2] = image_width - cx
    return K_new


# -----------------------------------------------------------------------------
# View ranking by camera-pose similarity
# -----------------------------------------------------------------------------


def rank_views_by_pose_similarity(
    start_idx: int,
    extrinsics_c2w: np.ndarray,
    lambda_t: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Return view indices sorted by pose-distance from ``start_idx``.

    The distance is ``r_dist + lambda_t * t_dist`` where ``r_dist`` is the SO(3)
    geodesic angle (normalized to ``[0, 1]`` via ``angle / pi``) and ``t_dist``
    is Euclidean translation distance (optionally normalized so its mean
    magnitude is ``1`` before measuring distances).
    """
    translations = extrinsics_c2w[:, :3, 3]
    if normalize:
        mean_norm = float(np.mean(np.linalg.norm(translations, axis=1)))
        if mean_norm > 0:
            translations = translations / mean_norm

    t_dist = np.linalg.norm(translations - translations[start_idx], axis=1)
    r_dist_rad = so3_relative_angle(extrinsics_c2w[:, :3, :3], extrinsics_c2w[start_idx, None, :3, :3])
    r_dist = np.rad2deg(r_dist_rad) / 180.0

    return np.argsort(r_dist + lambda_t * t_dist)


# -----------------------------------------------------------------------------
# Depth thresholding (per-frame outlier suppression)
# -----------------------------------------------------------------------------


def clip_depth_outliers(
    depth_map: np.ndarray,
    max_percentile: int = 0,
    min_percentile: int = 0,
    max_depth: Optional[float] = None,
) -> np.ndarray:
    """Zero out invalid / outlier depth values.

    Args:
        depth_map: ``(H, W)`` depth, modified in place. Invalid pixels are
            set to ``0``.
        max_percentile: when > 0, zero pixels above the ``max_percentile``-th
            percentile *of currently-valid pixels*.
        min_percentile: when > 0, zero pixels below the ``min_percentile``-th
            percentile *of currently-valid pixels*.
        max_depth: when not None, zero pixels above this absolute value first.

    Returns:
        The same array, mutated.
    """
    if max_depth is not None:
        depth_map[depth_map > max_depth] = 0.0

    valid = depth_map[depth_map > 0]
    if valid.size == 0:
        return depth_map

    if max_percentile > 0:
        hi = float(np.percentile(valid, max_percentile))
        if hi > 0:
            depth_map[depth_map > hi] = 0.0
    if min_percentile > 0:
        lo = float(np.percentile(valid, min_percentile))
        if lo > 0:
            depth_map[depth_map < lo] = 0.0

    return depth_map
