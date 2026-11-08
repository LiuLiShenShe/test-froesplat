# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in
# LICENSES/VGGT-LICENSE.txt in the root of this source tree.
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Camera pose & intrinsics supervision loss."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import huber_loss

from dvlt.common.constants import DataField
from dvlt.common.pose import inverse_pose
from dvlt.common.rays import s2_logmap_at_z1
from dvlt.model_components.loss.base import (
    BATCH_NORMALIZATION_SCALE_KEY,
    PRED_NORMALIZATION_SCALE_KEY,
)
from dvlt.model_components.pose_encoding import extri_intri_to_pose_enc

from .common import quaternion_loss, rotation_loss
from .util import check_and_fix_inf_nan


class CameraLoss(nn.Module):
    """Unified camera pose and intrinsics supervision loss.

    Supports two prediction formats:
    - "pose_enc": Pose encodings with shape [..., 7] or [..., 9]
    - "matrix": Raw 4x4 extrinsic matrices with shape [..., 4, 4]

    Supports deep supervision with lists of predictions.

    Returns per-sample losses of shape [B] for compatibility with
    sample_weights in MultiTaskLoss.
    """

    def __init__(
        self,
        pred_format: Literal["pose_enc", "matrix"] = "pose_enc",
        loss_type: str = "l1",
        gamma: float = 0.6,
        weight_T: float = 1.0,
        weight_R: float = 1.0,
        weight_fl: float = 0.5,
        weight_rays: float = 0.5,
        weight_fov: float = 0.5,
        double_cover_aware: bool = False,
        pred_key: str = "pose_enc",
        pose_convention: Literal["c2w", "w2c"] = "c2w",
    ):
        """Initialize CameraLoss.

        Args:
            pred_format: "pose_enc" for pose encodings, "matrix" for 4x4 extrinsic matrices.
            loss_type: Loss type - "l1", "l2", or "huber" (pose encoding mode only).
            gamma: Deep supervision weight decay for multiple predictions.
            weight_T: Weight for translation loss.
            weight_R: Weight for rotation loss.
            weight_fl: Weight for focal length loss (pose encoding mode only).
            weight_rays: Weight for rays loss (matrix mode, if rays are predicted).
            weight_fov: Weight for fov loss (matrix mode, if fov is predicted).
            double_cover_aware: Use double cover aware loss for rotation (pose encoding mode only).
            pred_key: Key in predictions dict for pose predictions.
            pose_convention: Pose convention for ground truth computation - "c2w" or "w2c". Default: "c2w".
        """
        super().__init__()
        self.pred_format = pred_format
        self.loss_type = loss_type
        self.gamma = gamma
        self.weight_T = weight_T
        self.weight_R = weight_R
        self.weight_fl = weight_fl
        self.weight_rays = weight_rays
        self.weight_fov = weight_fov
        self.double_cover_aware = double_cover_aware
        self.pred_key = pred_key
        self.pose_convention = pose_convention

        if self.pred_format == "matrix" and self.pose_convention != "c2w":
            raise AssertionError("Matrix format only supports c2w pose convention")

    def check_inputs(self, predictions: Dict[str, Any], batch: Dict[str, Any]) -> bool:
        if self.pred_key not in predictions or predictions[self.pred_key] is None:
            return False
        required = [DataField.EXTRINSICS_C2W, DataField.POINT_MASKS]
        return all(key in batch for key in required)

    def _to_list(self, pred: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
        """Convert prediction to list for uniform handling of single vs multiple predictions."""
        if isinstance(pred, torch.Tensor):
            return [pred]
        return pred

    def forward(
        self,
        predictions: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_pose = predictions[self.pred_key]
        pred_pose_list = self._to_list(pred_pose)

        if self.pred_format == "matrix":
            return self._forward_matrix(predictions, batch, pred_pose_list)
        else:
            return self._forward_pose_enc(predictions, batch, pred_pose_list)

    def _forward_pose_enc(
        self,
        predictions: Dict[str, Any],
        batch: Dict[str, Any],
        pred_pose_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for pose encoding format.

        Returns per-sample losses of shape [B].
        """
        # Compute ground truth pose encoding
        H, W = batch[DataField.IMAGES].shape[-2:]

        # Get extrinsics in the correct pose convention
        gt_extrinsics = batch[DataField.EXTRINSICS_C2W]
        if self.pose_convention == "w2c":
            gt_extrinsics = inverse_pose(gt_extrinsics)

        gt_pose_enc = extri_intri_to_pose_enc(
            gt_extrinsics,
            batch[DataField.INTRINSICS],
            (H, W),
        )
        mask_valid = batch[DataField.POINT_MASKS]
        B = mask_valid.shape[0]

        # Get normalization scales if available
        pred_scale = predictions.get(PRED_NORMALIZATION_SCALE_KEY)
        batch_scale = batch.get(BATCH_NORMALIZATION_SCALE_KEY)

        # Normalize GT translations once (before loop) - inplace ok, no gradients
        if batch_scale is not None:
            gt_pose_enc[..., :3] = gt_pose_enc[..., :3] / batch_scale.view(-1, 1, 1)

        # Multi-stage supervision
        batch_valid_mask = mask_valid[:, 0].sum(dim=[-1, -2]) > 100  # (B,)
        num_predictions = len(pred_pose_list)

        loss_T = torch.zeros(B, device=mask_valid.device)
        loss_R = torch.zeros(B, device=mask_valid.device)
        loss_fl = torch.zeros(B, device=mask_valid.device)

        for i in range(num_predictions):
            i_weight = self.gamma ** (num_predictions - i - 1)
            cur_pred = pred_pose_list[i]  # (B, S, D)

            # Apply normalization to predicted translations
            if pred_scale is not None:
                pred_translations = cur_pred[..., :3] / pred_scale.view(-1, 1, 1)
                cur_pred = torch.cat([pred_translations, cur_pred[..., 3:]], dim=-1)

            if batch_valid_mask.sum() == 0:
                # Preserve gradient flow through predictions for DDP compatibility
                safe_pred = torch.where(torch.isfinite(cur_pred), cur_pred, torch.zeros_like(cur_pred))
                loss_T_i = (safe_pred * 0).reshape(B, -1).mean(dim=1)  # [B]
                loss_R_i = loss_T_i
                loss_fl_i = loss_T_i
            else:
                loss_T_i, loss_R_i, loss_fl_i = self._forward_single_pose_enc(
                    cur_pred[batch_valid_mask],
                    gt_pose_enc[batch_valid_mask],
                )
                # Scatter per-sample losses back to full batch
                loss_T_i = _scatter_to_batch(loss_T_i, batch_valid_mask, B, cur_pred)
                loss_R_i = _scatter_to_batch(loss_R_i, batch_valid_mask, B, cur_pred)
                loss_fl_i = _scatter_to_batch(loss_fl_i, batch_valid_mask, B, cur_pred)

            loss_T = loss_T + loss_T_i * i_weight
            loss_R = loss_R + loss_R_i * i_weight
            loss_fl = loss_fl + loss_fl_i * i_weight

        loss_T = loss_T / num_predictions
        loss_R = loss_R / num_predictions
        loss_fl = loss_fl / num_predictions
        loss = loss_T * self.weight_T + loss_R * self.weight_R + loss_fl * self.weight_fl  # [B]

        loss_dict = {"loss_T": loss_T.mean(), "loss_R": loss_R.mean()}
        if gt_pose_enc.shape[-1] == 9:
            loss_dict["loss_fl"] = loss_fl.mean()
        return loss, loss_dict

    def _forward_matrix(
        self,
        predictions: Dict[str, Any],
        batch: Dict[str, Any],
        pred_pose_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for 4x4 matrix format.

        Returns per-sample losses of shape [B].
        """
        batch_valid_mask = batch[DataField.POINT_MASKS][:, 0].sum(dim=[-1, -2]) > 100  # (B,)
        cam_gt = batch[DataField.EXTRINSICS_C2W]
        B = cam_gt.shape[0]
        num_predictions = len(pred_pose_list)

        # Get normalization scales if available
        pred_scale = predictions.get(PRED_NORMALIZATION_SCALE_KEY)
        batch_scale = batch.get(BATCH_NORMALIZATION_SCALE_KEY)

        # Normalize GT translations once (before loop)
        if batch_scale is not None:
            cam_gt = cam_gt.clone()
            cam_gt[..., :3, 3] = cam_gt[..., :3, 3] / batch_scale.view(-1, 1, 1)

        loss_T = torch.zeros(B, device=cam_gt.device)
        loss_R = torch.zeros(B, device=cam_gt.device)

        for i in range(num_predictions):
            i_weight = self.gamma ** (num_predictions - i - 1)
            cam_pred = pred_pose_list[i]

            # Apply normalization to predicted translations (avoid inplace for gradients)
            if pred_scale is not None:
                pred_T = cam_pred[..., :3, 3:] / pred_scale.view(-1, 1, 1, 1)
                cam_pred = torch.cat([cam_pred[..., :3, :3], pred_T], dim=-1)
                cam_pred = torch.cat(
                    [cam_pred, cam_pred.new_tensor([0, 0, 0, 1]).expand(*cam_pred.shape[:-2], 1, 4)], dim=-2
                )

            if batch_valid_mask.sum() == 0:
                safe_pred = torch.where(torch.isfinite(cam_pred), cam_pred, torch.zeros_like(cam_pred))
                loss_T_i = (safe_pred * 0.0).reshape(B, -1).mean(dim=1)  # [B]
                loss_R_i = loss_T_i
            else:
                loss_T_i, loss_R_i = self._forward_single_matrix(
                    cam_pred[batch_valid_mask],
                    cam_gt[batch_valid_mask],
                )
                loss_T_i = _scatter_to_batch(loss_T_i, batch_valid_mask, B, cam_pred)
                loss_R_i = _scatter_to_batch(loss_R_i, batch_valid_mask, B, cam_pred)

            loss_T = loss_T + loss_T_i * i_weight
            loss_R = loss_R + loss_R_i * i_weight

        loss_T = loss_T / num_predictions
        loss_R = loss_R / num_predictions
        loss = loss_T * self.weight_T + loss_R * self.weight_R  # [B]

        loss_dict = {"loss_T": loss_T.mean(), "loss_R": loss_R.mean()}

        # Optional rays loss (per-sample)
        if "rays" in predictions:
            H, W = batch[DataField.IMAGES].shape[-2:]
            gt_tangent_coords = _tangent_coords_from_intrinsics(batch[DataField.INTRINSICS], H, W)
            loss_rays = (predictions["tangent_coords"] - gt_tangent_coords).abs().reshape(B, -1).mean(dim=1)  # [B]
            loss_dict["loss_rays"] = loss_rays.mean()
            loss = loss + self.weight_rays * loss_rays
        # Optional fov loss (per-sample)
        elif "fov" in predictions:
            H, W = batch[DataField.IMAGES].shape[-2:]
            fov_h = 2 * torch.atan((H / 2) / batch[DataField.INTRINSICS][..., 1, 1])
            fov_w = 2 * torch.atan((W / 2) / batch[DataField.INTRINSICS][..., 0, 0])
            loss_fov = (predictions["fov"] - torch.stack([fov_h, fov_w], dim=-1)).abs().reshape(B, -1).mean(dim=1)
            loss_dict["loss_fov"] = loss_fov.mean()
            loss = loss + self.weight_fov * loss_fov

        return loss, loss_dict

    def _forward_single_pose_enc(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample loss for a single pose encoding prediction.

        Args:
            pred: (N, S, D) predictions for valid batch elements
            gt: (N, S, D) ground truth for valid batch elements

        Returns:
            Tuple of per-sample losses (loss_T, loss_R, loss_fl), each of shape [N].
        """
        if self.loss_type == "l1":
            loss_fn = lambda x, y: (x - y).abs()
        elif self.loss_type == "l2":
            loss_fn = lambda x, y: (x - y).norm(dim=-1, keepdim=True)
        elif self.loss_type == "huber":
            loss_fn = lambda x, y: huber_loss(x, y, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        N = pred.shape[0]

        loss_T = loss_fn(pred[..., :3], gt[..., :3])  # (N, S, 3) or (N, S, 1)
        if self.double_cover_aware:
            loss_R = quaternion_loss(pred[..., 3:7], gt[..., 3:7], loss_fn=loss_fn)  # (N, S, 4)
        else:
            loss_R = loss_fn(pred[..., 3:7], gt[..., 3:7])  # (N, S, 4)

        if gt.shape[-1] == 9:
            loss_fl = loss_fn(pred[..., 7:], gt[..., 7:])  # (N, S, 2)
        else:
            loss_fl = torch.zeros(N, device=gt.device)

        loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
        loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
        loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

        # Reduce to per-sample: mean over S and feature dims
        loss_T = loss_T.clamp(max=100).reshape(N, -1).mean(dim=1)  # [N]
        loss_R = loss_R.reshape(N, -1).mean(dim=1)  # [N]
        if loss_fl.dim() > 1:
            loss_fl = loss_fl.reshape(N, -1).mean(dim=1)  # [N]

        return loss_T, loss_R, loss_fl

    def _forward_single_matrix(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-sample loss for a single 4x4 matrix prediction.

        Args:
            pred: (N, S, 4, 4) predicted extrinsics for valid batch elements
            gt: (N, S, 4, 4) ground truth extrinsics for valid batch elements

        Returns:
            Tuple of per-sample losses (loss_T, loss_R), each of shape [N].
        """
        # Per-sample L1 for translation
        loss_T = (pred[..., :3, 3] - gt[..., :3, 3]).abs().reshape(pred.shape[0], -1).mean(dim=1)  # [N]
        # Per-sample geodesic for rotation
        loss_R = rotation_loss(pred[..., :3, :3], gt[..., :3, :3]).reshape(pred.shape[0], -1).mean(dim=1)  # [N]
        return loss_T, loss_R


def _scatter_to_batch(
    valid_losses: torch.Tensor,
    valid_mask: torch.Tensor,
    B: int,
    pred: torch.Tensor,
) -> torch.Tensor:
    """Scatter per-valid-sample losses back to full batch, preserving gradients.

    Invalid samples get a zero-valued loss connected to predictions for DDP grad sync.

    Args:
        valid_losses: (N,) losses for valid samples
        valid_mask: (B,) boolean mask indicating which samples are valid
        B: total batch size
        pred: prediction tensor to connect invalid-sample zeros to (for gradient flow)

    Returns:
        (B,) losses with valid losses placed and zeros for invalid samples
    """
    safe_pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))
    result = (safe_pred * 0).reshape(B, -1).mean(dim=1)  # [B] zeros connected to pred graph
    result[valid_mask] = valid_losses
    return result


def _tangent_coords_from_intrinsics(intrinsics: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert intrinsics to S² tangent coordinates by generating rays for each pixel.

    Fully vectorized: builds the GT per-pixel unit-ray field from ``K`` and projects
    each ray onto the tangent plane at ``(0, 0, 1)`` via ``s2_logmap_at_z1``. Pixel
    centers use the ``integer + 0.5`` convention to match the data-loading pipeline.
    """
    B, S, _, _ = intrinsics.shape
    device = intrinsics.device
    dtype = intrinsics.dtype

    # Create pixel coordinate grid: (H, W, 2) with [u, v] coordinates
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    pixel_coords = torch.stack([u_coords, v_coords], dim=-1)  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5  # Add 0.5 to get pixel centers

    # Extract intrinsic parameters: (B, S)
    fx = intrinsics[..., 0, 0]  # (B, S)
    fy = intrinsics[..., 1, 1]  # (B, S)
    cx = intrinsics[..., 0, 2]  # (B, S)
    cy = intrinsics[..., 1, 2]  # (B, S)

    # Broadcast pixel coordinates to match batch dimensions: (B, S, H, W)
    u = pixel_coords[None, None, ..., 0]  # (1, 1, H, W)
    v = pixel_coords[None, None, ..., 1]  # (1, 1, H, W)

    # Unproject pixels to rays: ray = normalize([(u - cx)/fx, (v - cy)/fy, 1])
    # Reshape intrinsic params for broadcasting: (B, S, 1, 1)
    fx = fx[..., None, None]
    fy = fy[..., None, None]
    cx = cx[..., None, None]
    cy = cy[..., None, None]

    # Compute unnormalized ray directions: (B, S, H, W, 3)
    ray_x = (u - cx) / fx
    ray_y = (v - cy) / fy

    # Stack and normalize: (B, S, H, W, 3)
    rays = torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)
    rays = F.normalize(rays, dim=-1)

    # Convert rays to tangent coordinates at z1: (B, S, H, W, 2)
    tangent_coords = s2_logmap_at_z1(rays)
    return tangent_coords
