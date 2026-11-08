# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Pose and intrinsics evaluators."""

from dataclasses import dataclass

import torch

from dvlt.callbacks.base import CallbackConfig
from dvlt.callbacks.evaluators.base import Evaluator
from dvlt.common.constants import DataField, PredictionField
from dvlt.common.pose import to4x4
from dvlt.metric.intrinsic import compute_intrinsics_metrics
from dvlt.metric.pose import compute_pose_metrics


@dataclass
class PoseEvaluatorConfig(CallbackConfig):
    """Pose evaluator configuration."""

    _target_: str = "dvlt.callbacks.evaluators.pose.PoseEvaluator"
    pose_auc_thresholds: tuple[int, ...] = (1, 3, 5, 20, 30)


class PoseEvaluator(Evaluator):
    """Evaluator for camera pose (rotation and translation).

    Reports standard pose metrics: AUC_1, AUC_5, AUC_20, and per-threshold
    rotation/translation accuracies + mean errors.
    """

    METRIC_KEYS = (
        "R_error",
        "T_error",
        "Racc_1",
        "Tacc_1",
        "Auc_1",
        "Racc_5",
        "Tacc_5",
        "Auc_5",
        "Racc_20",
        "Tacc_20",
        "Auc_20",
    )

    def __init__(
        self,
        *args,
        pose_auc_thresholds: tuple[int, ...] = (1, 3, 5, 20, 30),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pose_auc_thresholds = pose_auc_thresholds

    def _compute_batch_metrics(self, batch: dict, predictions: dict, output_dir: str, batch_idx: int) -> dict:
        """Compute pose metrics for a batch."""
        cameras_pred = predictions.get(PredictionField.CAMERAS, None)
        extrinsics_c2w_gt = batch.get(DataField.EXTRINSICS_C2W, None)

        if cameras_pred is None or extrinsics_c2w_gt is None:
            device = batch[DataField.IMAGES].device
            nan = torch.tensor(float("nan"), device=device, dtype=torch.float32)
            return {k: nan for k in self.METRIC_KEYS}

        pred_c2w = to4x4(cameras_pred.camera_to_worlds)
        gt_c2w = to4x4(extrinsics_c2w_gt)

        with torch.autocast("cuda", dtype=torch.float64):
            return compute_pose_metrics(
                pred_c2w.to(torch.float64), gt_c2w.to(torch.float64), thresholds=self.pose_auc_thresholds
            )

    def _get_table_title(self) -> str:
        return "Pose Metrics"


@dataclass
class IntrinsicsEvaluatorConfig(CallbackConfig):
    """Intrinsics evaluator configuration."""

    _target_: str = "dvlt.callbacks.evaluators.pose.IntrinsicsEvaluator"


class IntrinsicsEvaluator(Evaluator):
    """Evaluator for camera intrinsics."""

    METRIC_KEYS = ("focal_x_error", "focal_y_error", "focal_mean_error")

    def _compute_batch_metrics(self, batch: dict, predictions: dict, output_dir: str, batch_idx: int) -> dict:
        """Compute intrinsics metrics for a batch."""
        cameras_pred = predictions.get(PredictionField.CAMERAS, None)
        intrinsics_gt = batch.get(DataField.INTRINSICS, None)

        if cameras_pred is None or intrinsics_gt is None:
            device = batch[DataField.IMAGES].device
            nan = torch.tensor(float("nan"), device=device, dtype=torch.float32)
            return {k: nan for k in self.METRIC_KEYS}

        intrinsics_pred = cameras_pred.get_intrinsics_matrices()
        return compute_intrinsics_metrics(intrinsics_pred, intrinsics_gt)

    def _get_table_title(self) -> str:
        return "Intrinsics Metrics"
