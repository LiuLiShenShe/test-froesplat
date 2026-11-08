# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Utility functions for callbacks (preprocessing, alignment, etc.)."""

import torch
from torch import Tensor

from dvlt.common.constants import DataField, PredictionField
from dvlt.common.geometry import transform_points, umeyama_alignment
from dvlt.model_components.loss.util import torch_quantile
from dvlt.struct.util import extri_intri_to_cameras


def index_batch_and_predictions(
    batch: dict,
    predictions: dict,
    batch_idx: int,
    seq_idxs: Tensor | None = None,
    inplace: bool = False,
) -> tuple[dict, dict]:
    """Return (batch, predictions) indexed at batch_idx and optionally subsequenced.

    Args:
        batch: Input batch dictionary
        predictions: Input predictions dictionary
        batch_idx: Index to select from batch dimension
        seq_idxs: Optional sequence indices for subsequencing
        inplace: If True, modifies input dicts in-place. If False, creates copies (default).

    Returns:
        Tuple of (batch, predictions) dictionaries, either modified copies or the original dicts.

    - Cast float tensors to float32
    - Detach prediction tensors
    - Cameras are carried through as-is for the selected batch
    """
    # Fields with sequence dimension
    batch_sequence_fields = {
        DataField.IDS,
        DataField.ORIGINAL_SIZES,
        DataField.IMAGES,
        DataField.DEPTHS,
        DataField.EXTRINSICS_C2W,
        DataField.INTRINSICS,
        DataField.WORLD_POINTS,
        DataField.POINT_MASKS,
    }
    pred_sequence_fields = {
        PredictionField.CAMERAS,
        PredictionField.DEPTHS,
        PredictionField.DEPTHS_CONF,
        PredictionField.WORLD_POINTS,
        PredictionField.WORLD_POINTS_DIRECT,
        PredictionField.WORLD_POINTS_DIRECT_CONF,
    }

    # Use original dicts or create shallow copies based on inplace flag
    if inplace:
        batch_out = batch
        predictions_out = predictions
    else:
        batch_out = dict(batch)
        predictions_out = dict(predictions)

    # Index batch dimension for batch fields (no scaling here)
    for field in DataField:
        if field not in batch or batch[field] is None:
            continue
        indexed = batch[field][batch_idx]
        if isinstance(indexed, Tensor) and indexed.dtype in (torch.float16, torch.float32):
            indexed = indexed.to(torch.float32)
        batch_out[field] = indexed

    # Index batch dimension for prediction fields
    for field in PredictionField:
        if field not in predictions or predictions[field] is None:
            continue
        indexed = predictions[field][batch_idx]
        if isinstance(indexed, Tensor):
            indexed = indexed.detach()
            if indexed.dtype in (torch.float16, torch.float32):
                indexed = indexed.to(torch.float32)
        predictions_out[field] = indexed

    # Extract camera matrices for depth unprojection
    cameras_pred = predictions_out[PredictionField.CAMERAS]
    pred_extrinsic_c2w = cameras_pred.camera_to_worlds.detach().to(torch.float32)
    pred_intrinsic = cameras_pred.get_intrinsics_matrices().detach().to(torch.float32)

    # Sequence subsampling
    if seq_idxs is not None:
        for field, tensor in list(batch_out.items()):
            if field in batch_sequence_fields:
                batch_out[field] = tensor[seq_idxs]
        for field, tensor in list(predictions_out.items()):
            if field in pred_sequence_fields:
                predictions_out[field] = tensor[seq_idxs]
        pred_extrinsic_c2w = pred_extrinsic_c2w[seq_idxs]
        pred_intrinsic = pred_intrinsic[seq_idxs]

    predictions_out[PredictionField.CAMERAS] = extri_intri_to_cameras(
        pred_extrinsic_c2w, pred_intrinsic, batch_out[DataField.IMAGES].shape[-2:]
    )
    return batch_out, predictions_out


def scale_batch_fields(batch: dict, inplace: bool = True) -> dict:
    """Scale batch dict by `DataField.SCALE_FACTOR`.

    Expects and indexed batch and predictions pair (i.e. after index_batch_and_predictions).

    Scales translations in extrinsics; scales full tensors for depths and world points.
    """
    if inplace:
        batch_out = batch
    else:
        batch_out = dict(batch)

    if DataField.SCALE_FACTOR not in batch:
        return batch_out

    scale_factor = batch_out[DataField.SCALE_FACTOR]
    batch_scale_fields = {
        DataField.DEPTHS,
        DataField.WORLD_POINTS,
        DataField.EXTRINSICS_C2W,
    }
    for field in DataField:
        if field not in batch_out or batch_out[field] is None:
            continue
        value = batch_out[field]
        if not isinstance(value, Tensor) or value.dtype not in (torch.float16, torch.float32):
            continue
        if field in batch_scale_fields:
            if field == DataField.EXTRINSICS_C2W:
                scaled_t = value[..., :3, 3] * scale_factor
                value_scaled = torch.cat([value[..., :3, :3], scaled_t[..., None]], dim=-1)
                batch_out[field] = value_scaled
            else:
                batch_out[field] = value * scale_factor

    return batch_out


def align_predictions_to_gt(
    indexed_batch: dict,
    indexed_predictions: dict,
    use_transform: bool = True,
    use_scale: bool = True,
    inplace: bool = False,
    outlier_rejection_iters: int = 1,
    outlier_rejection_percentile: float = 95.0,
) -> dict:
    """Return new (batch, predictions) with scale, SE3, Sim3 alignment applied to predictions.

    Expects and indexed batch and predictions pair (i.e. after index_batch_and_predictions).

    - If GT world points + mask are present: aligns cameras, scales depth, transforms 3D preds
    - Adds `transform` and `scale_factor` to predictions

    Args:
        outlier_rejection_iters: Number of iterative refinement rounds. After each Umeyama
            solve, points with residual above ``outlier_rejection_percentile`` are discarded
            and the alignment is re-computed on the inlier set. 0 = no rejection (default 1).
        outlier_rejection_percentile: Percentile threshold (0-100) for residual-based
            outlier rejection. Points above this percentile are discarded each round.
    """
    if inplace:
        preds_out = indexed_predictions
    else:
        preds_out = dict(indexed_predictions)

    # Use existing world points (should be computed from depth beforehand)
    world_pred = preds_out[PredictionField.WORLD_POINTS]

    world_gt = indexed_batch.get(DataField.WORLD_POINTS, None)
    point_mask = indexed_batch.get(DataField.POINT_MASKS, None)

    if world_gt is not None and point_mask is not None:
        pts_pred = world_pred[point_mask]  # (N, 3)
        pts_gt = world_gt[point_mask]  # (N, 3)

        rot, trans, scale = umeyama_alignment(pts_pred.T, pts_gt.T, with_scale=use_scale)

        for _ in range(outlier_rejection_iters):
            cur_transform = torch.cat([rot, trans[..., None]], dim=-1)
            aligned = transform_points(scale * pts_pred, cur_transform)
            residuals = (aligned - pts_gt).norm(dim=-1)
            threshold = torch_quantile(residuals, outlier_rejection_percentile / 100.0)
            inlier_mask = residuals <= threshold
            if inlier_mask.sum() < 10:
                break
            rot, trans, scale = umeyama_alignment(pts_pred[inlier_mask].T, pts_gt[inlier_mask].T, with_scale=use_scale)

        transform = torch.cat([rot, trans[..., None]], dim=-1)
        if not use_transform:
            transform = torch.eye(4, device=world_pred.device, dtype=torch.float32)

        # Update world points
        preds_out[PredictionField.WORLD_POINTS] = transform_points(scale * world_pred, transform)

        # Scale depths
        if PredictionField.DEPTHS in preds_out and preds_out[PredictionField.DEPTHS] is not None:
            preds_out[PredictionField.DEPTHS] = preds_out[PredictionField.DEPTHS] * scale

        if (
            PredictionField.WORLD_POINTS_DIRECT in preds_out
            and preds_out[PredictionField.WORLD_POINTS_DIRECT] is not None
        ):
            world_pred_direct = preds_out[PredictionField.WORLD_POINTS_DIRECT]
            preds_out[PredictionField.WORLD_POINTS_DIRECT] = transform_points(scale * world_pred_direct, transform)

        # TODO: make this a predictionfield?
        if "motion" in preds_out:
            motion_pred = preds_out["motion"].permute(0, 1, 3, 4, 2)
            # because motion is an offset, we don't need to add the translation
            preds_out["motion"] = ((scale * (motion_pred @ transform[:3, :3].T))).permute(0, 1, 4, 2, 3)

        # TODO: make this a predictionfield?
        if "motion_all" in preds_out:
            motion_pred_all = preds_out["motion_all"].permute(0, 1, 2, 4, 5, 3)
            preds_out["motion_all"] = ((scale * (motion_pred_all @ transform[:3, :3].T))).permute(0, 1, 2, 5, 3, 4)

        # Transform camera poses
        cameras_pred = preds_out[PredictionField.CAMERAS]
        pred_extrinsic_c2w = cameras_pred.camera_to_worlds
        pred_intrinsic = cameras_pred.get_intrinsics_matrices()

        aligned_extr = pred_extrinsic_c2w.clone()
        aligned_extr[:, :3, :3] = transform[:3, :3] @ aligned_extr[:, :3, :3]
        aligned_extr[:, :3, 3] = transform_points(scale * aligned_extr[:, :3, 3], transform)

        preds_out[PredictionField.CAMERAS] = extri_intri_to_cameras(
            aligned_extr, pred_intrinsic, indexed_batch[DataField.IMAGES].shape[-2:]
        )

        # Save transform info
        preds_out["transform"] = transform
        preds_out["scale_factor"] = float(scale)
        return preds_out

    # No alignment: keep everything as-is
    transform = torch.eye(4, device=world_pred.device, dtype=torch.float32)
    preds_out["transform"] = transform
    preds_out["scale_factor"] = 1.0
    return preds_out
