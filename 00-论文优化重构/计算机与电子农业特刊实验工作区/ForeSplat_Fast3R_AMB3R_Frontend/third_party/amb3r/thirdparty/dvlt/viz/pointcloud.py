# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Depth-unprojection + filtering pipeline that turns predictions into a cloud.

Two entry points:

* :func:`build_pointcloud` returns ``(pts, colors)`` — what the Gradio /
  rerun GLB exporters need.
* :func:`build_pointcloud_with_frame_ids` additionally returns per-point
  source-frame indices, for callers that need to know which input frame
  each surviving point originated from (e.g. cumulative reveal animations).

Pipeline (applied identically in both entries):

1. Depth-unproject each ``(S, H, W)`` depth into world coordinates using the
   predicted ``Cameras``.
2. Mask out invalid pixels: ``depth_to_world_coords_points``'s validity mask,
   pad pixels (when ``gradio_valid_pixels`` is provided), and sky pixels
   (via :func:`dvlt.util.skyseg.apply_sky_segmentation`).
3. Trim by confidence: prefer ``DEPTHS_CONF`` (aligned with depth
   unprojection) and fall back to ``WORLD_POINTS_DIRECT_CONF``. Drop points
   below the ``conf_threshold``-th percentile so the slider scales linearly
   regardless of the model's raw confidence distribution.
4. Optional spatial outlier trim by L2 distance from the cloud median.
5. Random subsample to ``max_points``.

All three filters apply consistently so any companion array (frame ids,
extra metadata) tracks ``pts``/``colors`` row-for-row.
"""

import logging
from typing import Optional

import numpy as np

from dvlt.common.constants import DataField, PredictionField
from dvlt.common.geometry import depth_to_world_coords_points
from dvlt.common.pose import to4x4
from dvlt.util.skyseg import apply_sky_segmentation


logger = logging.getLogger(__name__)


# Spatial trim is disabled in the UI (100 = off); kept as a code-level constant
# so build_pointcloud's existing kwarg still works.
SPATIAL_PERCENTILE_DEFAULT: float = 100.0


__all__ = [
    "SPATIAL_PERCENTILE_DEFAULT",
    "zero_depths_on_pad",
    "filter_spatial_outliers",
    "build_pointcloud",
    "build_pointcloud_with_frame_ids",
]


def zero_depths_on_pad(depths_np: np.ndarray, batch: dict) -> np.ndarray:
    """Set depth to 0 on synthetic pad pixels so depth overlays stay off black bars."""
    pad_ok = batch.get("gradio_valid_pixels", None)
    if pad_ok is None:
        return depths_np
    m = pad_ok[0].detach().cpu().numpy()
    out = depths_np.copy()
    out[~m] = 0.0
    return out


def filter_spatial_outliers(
    pts_np: np.ndarray,
    colors_np: np.ndarray,
    spatial_percentile: float,
    *,
    extras: Optional[list[np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Drop the farthest points from the cloud median (by L2 distance).

    Confidence filtering does not remove geometry spikes with moderately
    high confidence; this trims the tail of per-point distances from
    ``median(pts)``. A ``spatial_percentile`` of 100 disables; otherwise
    keep points whose distance-from-median is at or below that percentile
    of all distances (drops the upper tail).

    ``extras`` is an optional list of parallel arrays that get the same
    boolean mask applied row-for-row (e.g. frame ids).
    """
    extras = list(extras) if extras is not None else []
    if spatial_percentile >= 100.0 or len(pts_np) < 32:
        return pts_np, colors_np, extras
    center = np.median(pts_np, axis=0)
    dist = np.linalg.norm(pts_np - center, axis=1)
    d_cap = float(np.percentile(dist, spatial_percentile))
    keep = dist <= d_cap
    return pts_np[keep], colors_np[keep], [e[keep] for e in extras]


def _build_pointcloud_core(
    predictions: dict,
    batch: dict,
    max_points: int,
    conf_threshold: float,
    spatial_percentile: float,
    *,
    track_frame_ids: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Internal: full pipeline with optional per-point source-frame tracking."""
    depths = predictions[PredictionField.DEPTHS][0]  # (S, H, W)
    cameras = predictions[PredictionField.CAMERAS][0]
    extrinsics_c2w = to4x4(cameras.camera_to_worlds)  # (S, 4, 4)
    intrinsics = cameras.get_intrinsics_matrices()  # (S, 3, 3)

    world_points, _, valid_mask = depth_to_world_coords_points(
        depths,
        extrinsics_c2w,
        intrinsics,
    )

    images = batch[DataField.IMAGES][0]  # (S, C, H, W)
    # NumPy has no bfloat16 dtype; cast to float32 before .numpy() for models
    # (e.g. Pi3) that run inference in bfloat16.
    colors = images.detach().float().cpu().permute(0, 2, 3, 1).numpy() * 255.0

    S, _, H, W = images.shape
    pts_np = world_points.detach().float().cpu().numpy().reshape(-1, 3)
    colors_np = colors.reshape(-1, 3)
    mask_np = valid_mask.detach().cpu().numpy().reshape(-1)

    frame_ids_np: Optional[np.ndarray] = None
    if track_frame_ids:
        frame_ids_np = np.repeat(np.arange(S, dtype=np.int32), H * W)

    pad_ok = batch.get("gradio_valid_pixels", None)
    if pad_ok is not None:
        mask_np = mask_np & pad_ok[0].detach().cpu().numpy().reshape(-1)

    try:
        sky_conf = apply_sky_segmentation(
            np.ones((S, H, W), dtype=np.float32),
            images.detach().float().cpu(),
        )
        mask_np = mask_np & (sky_conf.reshape(-1) > 0.5)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Sky segmentation failed ({e}), skipping.")

    pts_np = pts_np[mask_np]
    colors_np = colors_np[mask_np]
    if frame_ids_np is not None:
        frame_ids_np = frame_ids_np[mask_np]

    confidence = predictions.get(PredictionField.DEPTHS_CONF, None)
    if confidence is None:
        confidence = predictions.get(PredictionField.WORLD_POINTS_DIRECT_CONF, None)

    if confidence is not None:
        conf_flat = confidence[0].detach().float().cpu().numpy().reshape(-1)
        conf_flat = conf_flat[mask_np]
        threshold_val = np.percentile(conf_flat, conf_threshold)
        keep = conf_flat >= threshold_val
        pts_np = pts_np[keep]
        colors_np = colors_np[keep]
        if frame_ids_np is not None:
            frame_ids_np = frame_ids_np[keep]

    extras = [frame_ids_np] if frame_ids_np is not None else None
    pts_np, colors_np, extras = filter_spatial_outliers(pts_np, colors_np, spatial_percentile, extras=extras)
    if extras:
        frame_ids_np = extras[0]

    effective_max = max_points if max_points > 0 else int(1e12)
    if len(pts_np) > effective_max:
        idx = np.random.choice(len(pts_np), effective_max, replace=False)
        pts_np = pts_np[idx]
        colors_np = colors_np[idx]
        if frame_ids_np is not None:
            frame_ids_np = frame_ids_np[idx]

    return pts_np, colors_np.astype(np.uint8), frame_ids_np


def build_pointcloud(
    predictions: dict,
    batch: dict,
    max_points: int,
    conf_threshold: float,
    spatial_percentile: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth-unproject predictions into a coloured point cloud.

    Always uses depth + predicted cameras for unprojection regardless of
    whether the model also predicts world points directly.

    Uses depth confidence when available (aligned with depth lifting), then
    falls back to direct world-point confidence. Optional spatial trimming
    drops points in the far tail of distance-from-median.

    Returns ``(points, colors)`` each shaped ``(N, 3)``. Colours are uint8.
    """
    pts, colors, _ = _build_pointcloud_core(
        predictions,
        batch,
        max_points,
        conf_threshold,
        spatial_percentile,
        track_frame_ids=False,
    )
    return pts, colors


def build_pointcloud_with_frame_ids(
    predictions: dict,
    batch: dict,
    max_points: int,
    conf_threshold: float,
    spatial_percentile: float = SPATIAL_PERCENTILE_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same pipeline as :func:`build_pointcloud` plus per-point source-frame ids.

    The third returned array, ``frame_ids`` of shape ``(N,)`` and dtype
    ``int32``, identifies the input frame each surviving point was
    unprojected from. Useful for any consumer that needs to attribute
    reconstructed geometry back to its source frame (e.g. cumulative reveal
    animations, per-frame point counts).
    """
    pts, colors, frame_ids = _build_pointcloud_core(
        predictions,
        batch,
        max_points,
        conf_threshold,
        spatial_percentile,
        track_frame_ids=True,
    )
    assert frame_ids is not None  # noqa: S101 — paired with track_frame_ids
    return pts, colors, frame_ids
