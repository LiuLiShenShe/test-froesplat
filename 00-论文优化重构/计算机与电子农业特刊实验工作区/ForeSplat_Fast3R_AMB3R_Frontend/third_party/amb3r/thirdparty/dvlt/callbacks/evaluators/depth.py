# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth evaluator."""

from dataclasses import dataclass

import torch

from dvlt.callbacks.base import CallbackConfig
from dvlt.callbacks.evaluators.base import Evaluator
from dvlt.common.constants import DataField, PredictionField
from dvlt.metric.depth import AlignMode, compute_depth_metrics


@dataclass
class DepthEvaluatorConfig(CallbackConfig):
    """Depth evaluator configuration."""

    _target_: str = "dvlt.callbacks.evaluators.depth.DepthEvaluator"
    align: AlignMode = "median"  # Alignment modes: median, scale, scale_and_shift, none


class DepthEvaluator(Evaluator):
    """Depth prediction evaluator (AbsRel, RMSE, Delta, etc.)."""

    METRIC_KEYS = ("AbsRel", "SqRel", "RMSE", "MAE", "Delta1", "Delta2", "Delta3")

    def __init__(self, *args, align: AlignMode = "median", **kwargs):
        super().__init__(*args, **kwargs)
        self.align = align

    def _compute_batch_metrics(self, batch: dict, predictions: dict, output_dir: str, batch_idx: int) -> dict:
        depth_pred = predictions.get(PredictionField.DEPTHS, None)
        depth_gt = batch.get(DataField.DEPTHS, None)
        point_mask = batch.get(DataField.POINT_MASKS, None)

        if depth_pred is None or depth_gt is None:
            device = batch[DataField.IMAGES].device
            nan = torch.tensor(float("nan"), device=device, dtype=torch.float32)
            return {k: nan for k in self.METRIC_KEYS}

        valid_mask = point_mask & (depth_gt > 0) if point_mask is not None else depth_gt > 0
        return compute_depth_metrics(depth_pred, depth_gt, valid_mask, align=self.align)

    def _get_table_title(self) -> str:
        return "Depth Metrics"
