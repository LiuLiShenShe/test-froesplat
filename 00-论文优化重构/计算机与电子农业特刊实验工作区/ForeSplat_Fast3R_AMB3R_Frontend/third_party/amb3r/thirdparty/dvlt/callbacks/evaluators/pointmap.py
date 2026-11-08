# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Pointcloud evaluator."""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
from torch import Tensor

from dvlt.callbacks.base import CallbackConfig
from dvlt.callbacks.evaluators.base import Evaluator
from dvlt.common.constants import DataField, PredictionField
from dvlt.common.geometry import icp_refine, voxel_downsample
from dvlt.metric.pointcloud import compute_point_cloud_metrics


@dataclass
class PointmapEvaluatorConfig(CallbackConfig):
    """Pointcloud evaluator configuration."""

    _target_: str = "dvlt.callbacks.evaluators.pointmap.PointmapEvaluator"
    compute_recon_metrics: bool = False
    f1_threshold: float = 0.05
    dataset_f1_thresholds: Dict[str, float] = field(default_factory=dict)
    voxel_size: float = 0.0078125
    dataset_voxel_sizes: Dict[str, float] = field(default_factory=dict)
    voxel_downsample: bool = True
    dataset_voxel_downsample: Dict[str, bool] = field(default_factory=dict)
    icp_threshold: float = 0.1  # 0 to disable
    dataset_icp_thresholds: Dict[str, float] = field(default_factory=dict)
    inlier_thresholds_cm: Tuple[float, ...] = (1.0, 5.0, 20.0)
    absrel_inlier_threshold: float = 0.03


class PointmapEvaluator(Evaluator):
    """Evaluator for 3D point map predictions.

    Computes L2 distance between predicted and ground truth 3D points.
    When ``compute_recon_metrics=True``, also computes NN-based acc/comp/F1.
    """

    METRIC_KEYS = ("l2_depth", "l2_world", "acc_mean", "acc_med", "comp_mean", "comp_med", "f1")

    def __init__(
        self,
        *args,
        compute_recon_metrics: bool = False,
        f1_threshold: float = 0.05,
        dataset_f1_thresholds: Dict[str, float] | None = None,
        voxel_size: float = 0.0078125,
        dataset_voxel_sizes: Dict[str, float] | None = None,
        voxel_downsample: bool = True,
        dataset_voxel_downsample: Dict[str, bool] | None = None,
        icp_threshold: float = 0.1,
        dataset_icp_thresholds: Dict[str, float] | None = None,
        inlier_thresholds_cm: Tuple[float, ...] = (1.0, 5.0, 20.0),
        absrel_inlier_threshold: float = 0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compute_recon_metrics = compute_recon_metrics
        self.f1_threshold = f1_threshold
        self.dataset_f1_thresholds = dataset_f1_thresholds or {}
        self.voxel_size = voxel_size
        self.dataset_voxel_sizes = dataset_voxel_sizes or {}
        self.voxel_downsample = voxel_downsample
        self.dataset_voxel_downsample = dataset_voxel_downsample or {}
        self.icp_threshold = icp_threshold
        self.dataset_icp_thresholds = dataset_icp_thresholds or {}
        self.inlier_thresholds_cm = inlier_thresholds_cm
        self.absrel_inlier_threshold = absrel_inlier_threshold
        self._dataset_name: str = ""

    def on_test_dataset_start(self, trainer, dataset_name: str):
        super().on_test_dataset_start(trainer, dataset_name)
        self._dataset_name = dataset_name

    def _compute_pointmap_metrics(
        self, world_pred_depth: Tensor, world_gt: Tensor, point_mask: Tensor, suffix: str = ""
    ) -> dict:
        pred = world_pred_depth[point_mask]
        gt = world_gt[point_mask]

        error_norm = torch.linalg.norm(pred - gt, dim=-1)
        gt_norm = torch.linalg.norm(gt, dim=-1)
        valid = gt_norm > 0
        absrel = error_norm[valid] / gt_norm[valid]

        result = {
            f"l2_{suffix}": error_norm.mean(),
            f"absrel_{suffix}": absrel.mean(),
            f"inlier_{suffix}@{self.absrel_inlier_threshold:.0%}": (absrel < self.absrel_inlier_threshold)
            .float()
            .mean(),
        }
        for t in self.inlier_thresholds_cm:
            result[f"inlier_{suffix}@{t:.0f}cm"] = (error_norm < t / 100.0).float().mean()
        return result

    def _compute_batch_metrics(self, batch: dict, predictions: dict, output_dir: str, batch_idx: int) -> dict:
        """Compute pointcloud metrics for a batch."""
        world_gt = batch.get(DataField.WORLD_POINTS, None)
        point_mask = batch.get(DataField.POINT_MASKS, None)
        world_pred_depth = predictions.get(PredictionField.WORLD_POINTS, None)
        world_pred_direct = predictions.get(PredictionField.WORLD_POINTS_DIRECT, None)

        device = batch[DataField.IMAGES].device
        nan = torch.tensor(float("nan"), device=device, dtype=torch.float32)

        if world_gt is None or point_mask is None:
            return {k: nan for k in self.METRIC_KEYS}

        result = {}
        # Points from depth (always available)
        if world_pred_depth is not None:
            result.update(self._compute_pointmap_metrics(world_pred_depth, world_gt, point_mask, suffix="depth"))

        # Direct 3D point predictions (if available)
        if world_pred_direct is not None:
            result.update(self._compute_pointmap_metrics(world_pred_direct, world_gt, point_mask, suffix="world"))

        if not self.compute_recon_metrics:
            return result

        if world_pred_depth is not None:
            pred_pts = world_pred_depth[point_mask]
            gt_pts = world_gt[point_mask]

            if self.dataset_voxel_downsample.get(self._dataset_name, self.voxel_downsample):
                vs = self.dataset_voxel_sizes.get(self._dataset_name, self.voxel_size)
                pred_pts = voxel_downsample(pred_pts, vs)
                gt_pts = voxel_downsample(gt_pts, vs)
            icp_thresh = self.dataset_icp_thresholds.get(self._dataset_name, self.icp_threshold)
            if icp_thresh > 0:
                pred_pts = icp_refine(pred_pts, gt_pts, threshold=icp_thresh)
            f1_thresh = self.dataset_f1_thresholds.get(self._dataset_name, self.f1_threshold)
            recon_metrics = compute_point_cloud_metrics(pred_pts, gt_pts, threshold=f1_thresh)
            result.update(
                {
                    "acc_mean": recon_metrics["acc_mean"],
                    "acc_med": recon_metrics["acc_med"],
                    "comp_mean": recon_metrics["comp_mean"],
                    "comp_med": recon_metrics["comp_med"],
                    f"f1@{f1_thresh * 100:.01f}cm": recon_metrics["f_score"],
                }
            )
        else:
            for k in ("acc_mean", "acc_med", "comp_mean", "comp_med", "f1"):
                result[k] = nan

        return result

    def _get_table_title(self) -> str:
        if self.compute_recon_metrics:
            if self.dataset_voxel_downsample.get(self._dataset_name, self.voxel_downsample):
                vs = self.dataset_voxel_sizes.get(self._dataset_name, self.voxel_size)
                return f"Pointmap Metrics (voxel={vs}m)"
            return "Pointmap Metrics (no voxel downsample)"
        return "Pointmap Metrics"
