# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization callbacks for model outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from accelerate.logging import get_logger
from accelerate.tracking import GeneralTracker
from PIL import Image

from dvlt.callbacks.base import Callback, CallbackConfig
from dvlt.callbacks.util import align_predictions_to_gt, index_batch_and_predictions, scale_batch_fields
from dvlt.common.constants import DataField, PredictionField
from dvlt.metric.depth import align_median
from dvlt.struct.util import extri_intri_to_cameras
from dvlt.viz.scene_plotly import scene_to_plotly
from dvlt.viz.scene_rerun import visualize_scene


matplotlib.use("Agg")

if TYPE_CHECKING:
    from dvlt.engine.trainer import Trainer


logger = get_logger(__name__)


def _exclude_top_l2_error_indices(
    pred_flat: np.ndarray,
    gt_flat: np.ndarray,
    valid_idx: np.ndarray,
    exclude_top_frac: float = 0.05,
) -> np.ndarray:
    """Remove ``valid_idx`` entries whose L2(pred−gt) lies in the worst ``exclude_top_frac`` of errors.

    Operates only on the subset indexed by ``valid_idx``. Used before random subsampling for vis.
    """
    if valid_idx.size == 0 or gt_flat is None:
        return valid_idx
    err = np.linalg.norm(pred_flat[valid_idx] - gt_flat[valid_idx], axis=1)
    thr = np.percentile(err, 100.0 * (1.0 - exclude_top_frac))
    return valid_idx[err <= thr]


@dataclass
class SceneVisualizationCallbackConfig(CallbackConfig):
    """Scene visualization callback configuration."""

    _target_: str = "dvlt.callbacks.visualization.SceneVisualizationCallback"
    max_points: int = 100_000
    log_to_wandb: bool = True
    save_rrd: bool = True
    log_train_every_n_steps: int = 500
    outlier_rejection_iters: int = 1
    outlier_rejection_percentile: float = 95.0


class SceneVisualizationCallback(Callback):
    """Callback for visualizing scene reconstruction results.

    This callback creates visualizations for both training and testing:
    - Point clouds (predicted vs ground truth)
    - Camera poses
    - 2D tracks (if available)
    - 3D scene in rerun format

    During training, only logs to wandb (no file saves).
    During testing, creates all visualizations and saves to disk.
    """

    def __init__(
        self,
        max_points: int = 100_000,
        log_to_wandb: bool = True,
        save_rrd: bool = True,
        log_train_every_n_steps: int = 500,
        outlier_rejection_iters: int = 1,
        outlier_rejection_percentile: float = 95.0,
    ):
        """Initialize visualization callback.

        Args:
            max_points: Maximum number of points to visualize.
            log_to_wandb: Whether to log visualizations to wandb.
            save_rrd: Whether to save rerun visualization files (test only).
            log_train_every_n_steps: Log training visualization every N steps.
            outlier_rejection_iters: Iterative Umeyama refinement rounds for Sim3 alignment.
            outlier_rejection_percentile: Residual percentile cutoff per rejection round.
        """
        self.max_points = max_points
        self.log_to_wandb = log_to_wandb
        self.save_rrd = save_rrd
        self.log_train_every_n_steps = log_train_every_n_steps
        self.outlier_rejection_iters = outlier_rejection_iters
        self.outlier_rejection_percentile = outlier_rejection_percentile

    def on_train_batch(
        self,
        batch: dict,
        predictions: dict,
        step: int,
        trainer: "Trainer",
    ) -> dict:
        """Log training-time scene visualization.

        Args:
            batch: Training batch.
            predictions: Model predictions (raw format).
            step: Current training step.
            trainer: Trainer instance.

        Returns:
            Empty dict (no metrics).
        """
        if not self.log_to_wandb or not trainer.accelerator.is_main_process:
            return {}

        # Only create expensive visualization every N*log_train_every_n_steps steps
        if step % self.log_train_every_n_steps != 0 or step == 0:
            return {}

        wandb_tracker = next((t for t in trainer.accelerator.trackers if t.name == "wandb"), None)
        if wandb_tracker is None:
            return {}

        # Postprocess predictions using model's method
        if hasattr(trainer.model, "_postprocess_predictions"):
            predictions = trainer.model._postprocess_predictions(batch, predictions)
        else:
            logger.warning("Model doesn't have _postprocess_predictions - skipping visualization")
            return {}

        # Check if predictions are postprocessed (have PredictionField keys)
        if PredictionField.CAMERAS not in predictions:
            logger.warning("Training predictions not postprocessed - skipping visualization")
            return {}

        # Index to first batch element (no sim3 alignment for train)
        batch_single, predictions_single = index_batch_and_predictions(
            batch, predictions, batch_idx=0, seq_idxs=None, inplace=False
        )

        images = batch_single[DataField.IMAGES]
        world_pred = predictions_single[PredictionField.WORLD_POINTS]
        world_gt = batch_single[DataField.WORLD_POINTS]
        point_mask = batch_single[DataField.POINT_MASKS]
        cameras_pred = predictions_single[PredictionField.CAMERAS]
        extrinsics_c2w_gt = batch_single[DataField.EXTRINSICS_C2W]
        intrinsics_gt = batch_single[DataField.INTRINSICS]
        cameras_gt = extri_intri_to_cameras(extrinsics_c2w_gt, intrinsics_gt, images.shape[-2:])
        seq_name = batch_single.get(DataField.SEQ_NAME, "train")

        # Prepare point clouds
        pts_pred, pts_gt, pred_rgb, gt_rgb = self._prepare_pointcloud_data(
            world_pred, images, world_gt, point_mask, max_points=self.max_points
        )

        # Create plotly figure
        fig = scene_to_plotly(
            seq_name, pts_pred, pred_rgb, cameras_pred, pts_gt, gt_rgb, cameras_gt, view_coordinates="RDF"
        )
        wandb_tracker.log({"train/scene_visualization": fig}, step=step)

        return {}

    def on_test_batch(
        self,
        batch: dict,
        predictions: dict,
        output_dir: str,
        batch_idx: int,
        trainer: "Trainer",
        trackers: List[GeneralTracker] = None,
        global_step: int = None,
        dataset_name: str = "",
    ) -> dict:
        """Create visualizations for a test batch.

        Args:
            batch: Ground truth batch.
            predictions: Model predictions.
            output_dir: Directory to save outputs.
            batch_idx: Index of the batch.
            trainer: Trainer instance.
            trackers: List of trackers (e.g., wandb).
            global_step: Global training step (for wandb logging).
            dataset_name: Name of the dataset.

        Returns:
            Empty dict (no metrics to log).
        """
        if not trainer.accelerator.is_main_process:
            return {}

        # Index to first batch element (no sim3 alignment for train)
        batch, predictions = index_batch_and_predictions(batch, predictions, batch_idx=0, seq_idxs=None, inplace=False)

        # 2) Scale GT fields to original scale
        batch = scale_batch_fields(batch, inplace=False)

        # 3) Sim3 alignment
        predictions = align_predictions_to_gt(
            batch,
            predictions,
            inplace=False,
            outlier_rejection_iters=self.outlier_rejection_iters,
            outlier_rejection_percentile=self.outlier_rejection_percentile,
        )

        seq_name = batch.get(DataField.SEQ_NAME, f"batch_{batch_idx:05d}")
        images = batch[DataField.IMAGES]
        world_gt = batch.get(DataField.WORLD_POINTS, None)
        point_mask = batch.get(DataField.POINT_MASKS, None)
        extrinsics_c2w_gt = batch.get(DataField.EXTRINSICS_C2W, None)
        intrinsics_gt = batch.get(DataField.INTRINSICS, None)

        # Get predictions
        cameras_pred = predictions[PredictionField.CAMERAS]
        world_pred = predictions[PredictionField.WORLD_POINTS]

        # Build GT cameras if available
        if extrinsics_c2w_gt is not None and intrinsics_gt is not None:
            cameras_gt = extri_intri_to_cameras(extrinsics_c2w_gt, intrinsics_gt, images.shape[-2:])
        else:
            cameras_gt = None

        # Prepare point clouds for visualization
        pts_pred, pts_gt, pred_rgb, gt_rgb = self._prepare_pointcloud_data(
            world_pred, images, world_gt, point_mask, max_points=self.max_points
        )

        # Log to wandb if tracker is available
        if self.log_to_wandb and trackers is not None and global_step is not None:
            wandb_tracker = next((tracker for tracker in trackers if tracker.name == "wandb"), None)
            if wandb_tracker is not None:
                fig = scene_to_plotly(
                    seq_name=seq_name,
                    pts_pred=pts_pred.reshape(-1, 3),
                    pts_gt=pts_gt.reshape(-1, 3) if pts_gt.size > 0 else pts_gt,
                    pred_rgb=pred_rgb.reshape(-1, 3),
                    gt_rgb=gt_rgb.reshape(-1, 3) if gt_rgb.size > 0 else gt_rgb,
                    cameras_pred=cameras_pred,
                    cameras_gt=cameras_gt,
                    view_coordinates="RDF",
                )
                wandb_tracker.log({f"val/{dataset_name}/scene_visualization": fig}, step=global_step)

        # Save rerun visualization
        if self.save_rrd:
            visualize_scene(
                seq_name,
                cameras_pred,
                pts_pred,
                pred_rgb,
                save_path=os.path.join(output_dir, f"{seq_name}.rrd"),
                view_coordinates="RDF",
            )

        return {}

    @staticmethod
    def _prepare_pointcloud_data(
        world_pred: torch.Tensor,
        images: torch.Tensor,
        world_gt: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
        max_points: int = 50_000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert dense tensors to subsampled coloured point clouds for visualisation.

        Args:
            world_pred: (S, H, W, 3) predicted world coordinates (torch, any device).
            images: (S, 3, H, W) RGB in [0,1] (torch, same device).
            world_gt: (S, H, W, 3) GT world coordinates (same shape / device).
            point_mask: (S, H, W) boolean mask of valid points (torch).
            max_points: subsample to at most this many points.

        Returns:
            pts_pred, pts_gt, pred_rgb, gt_rgb – all as NumPy arrays
        """
        color_flat = (images.detach().permute(0, 2, 3, 1).cpu().numpy() * 255.0).reshape(-1, 3)
        world_pred_flat = world_pred.detach().cpu().numpy().reshape(-1, 3)
        world_gt_flat = world_gt.detach().cpu().numpy().reshape(-1, 3) if world_gt is not None else None
        mask_flat = point_mask.detach().cpu().numpy().astype(bool).reshape(-1) if point_mask is not None else None

        if mask_flat is not None:
            valid_idx = np.where(mask_flat)[0]
        else:
            valid_idx = np.arange(world_pred_flat.shape[0])

        if world_gt_flat is not None and valid_idx.size > max_points:
            valid_idx = _exclude_top_l2_error_indices(world_pred_flat, world_gt_flat, valid_idx)
        if valid_idx.size > max_points:
            valid_idx = np.random.choice(valid_idx, max_points, replace=False)

        pts_pred = world_pred_flat[valid_idx, :]
        pts_gt = world_gt_flat[valid_idx, :] if world_gt_flat is not None else None
        colors = color_flat[valid_idx, :]

        pred_rgb = colors.astype(np.uint8)
        pts_gt = np.empty((0, 3)) if pts_gt is None else pts_gt
        gt_rgb = np.empty((0, 3)) if pts_gt is None else colors.astype(np.uint8)

        return pts_pred, pts_gt, pred_rgb, gt_rgb


def _render_depth_comparison(
    image: np.ndarray,
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: np.ndarray,
    cmap: str = "turbo",
) -> Image.Image:
    """Render a side-by-side depth comparison: image | pred | GT | error.

    Args:
        image: (H, W, 3) RGB in [0, 1].
        depth_pred: (H, W) predicted depth.
        depth_gt: (H, W) ground-truth depth.
        valid_mask: (H, W) boolean mask of valid pixels.
        cmap: Matplotlib colormap name.

    Returns:
        PIL Image with the four panels concatenated horizontally.
    """
    # Shared depth range from valid GT
    valid_gt = depth_gt[valid_mask]
    if valid_gt.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(valid_gt.min()), float(valid_gt.max())
        if vmin == vmax:
            vmax = vmin + 1.0

    colormap = plt.get_cmap(cmap)

    def depth_to_rgb(d: np.ndarray, do_mask=True) -> np.ndarray:
        normed = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
        colored = colormap(normed)[..., :3]  # (H, W, 3) float in [0, 1]
        if do_mask:
            colored[~valid_mask] = 0.0
        return colored

    pred_rgb = depth_to_rgb(depth_pred, do_mask=False)
    gt_rgb = depth_to_rgb(depth_gt)

    # Absolute error, normalised to [0, 1] by the depth range
    abs_error = np.abs(depth_pred - depth_gt) / (vmax - vmin)
    abs_error[~valid_mask] = 0.0
    error_rgb = plt.get_cmap("hot")(np.clip(abs_error, 0.0, 1.0))[..., :3]
    error_rgb[~valid_mask] = 0.0

    strip = np.concatenate([image, pred_rgb, gt_rgb, error_rgb], axis=1)
    strip = (np.clip(strip, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(strip)


@dataclass
class DepthVisualizationCallbackConfig(CallbackConfig):
    """Depth visualization callback configuration."""

    _target_: str = "dvlt.callbacks.visualization.DepthVisualizationCallback"
    log_train_every_n_steps: int = 500
    frame_idx: int = 0


class DepthVisualizationCallback(Callback):
    """Periodically logs a single depth comparison strip (image | pred | GT | error) to wandb.

    Works for both training (raw predictions) and validation (postprocessed predictions).
    """

    def __init__(self, log_train_every_n_steps: int = 500, frame_idx: int = 0):
        self.log_train_every_n_steps = log_train_every_n_steps
        self.frame_idx = frame_idx

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def on_train_batch(
        self,
        batch: dict,
        predictions: dict,
        step: int,
        trainer: "Trainer",
    ) -> dict:
        if not trainer.accelerator.is_main_process:
            return {}
        if step % self.log_train_every_n_steps != 0 or step == 0:
            return {}

        wandb_tracker = next((t for t in trainer.accelerator.trackers if t.name == "wandb"), None)
        if wandb_tracker is None:
            return {}

        # Raw predictions: depth is (B, S, H, W, 1)
        depth_pred = predictions.get("depth", None)
        depth_gt = batch.get(DataField.DEPTHS, None)
        images = batch.get(DataField.IMAGES, None)
        point_masks = batch.get(DataField.POINT_MASKS, None)

        if depth_pred is None or depth_gt is None or images is None:
            return {}

        b, s = 0, min(self.frame_idx, depth_pred.shape[1] - 1)
        img_np = images[b, s].detach().float().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        dp = depth_pred[b, s].detach().float().squeeze(-1)  # (H, W) tensor
        dg = depth_gt[b, s].detach().float()  # (H, W) tensor
        mask_t = point_masks[b, s].detach().bool() if point_masks is not None else (dg > 0)
        mask_t = mask_t & (dg > 0)

        dp = align_median(dp, dg, mask_t)

        mask = mask_t.cpu().numpy()
        dp = dp.cpu().numpy()
        dg = dg.cpu().numpy()

        pil = _render_depth_comparison(img_np, dp, dg, mask)
        wandb_tracker.log({"train/depth_vis": wandb.Image(pil, caption=f"step {step}")}, step=step)
        return {}

    # ------------------------------------------------------------------
    # Validation / Test
    # ------------------------------------------------------------------

    def on_test_batch(
        self,
        batch: dict,
        predictions: dict,
        output_dir: str,
        batch_idx: int,
        trainer: "Trainer",
        trackers: List[GeneralTracker] = None,
        global_step: int = None,
        dataset_name: str = "",
    ) -> dict:
        if not trainer.accelerator.is_main_process:
            return {}

        batch_single, preds_single = index_batch_and_predictions(
            batch,
            predictions,
            batch_idx=0,
            seq_idxs=None,
            inplace=False,
        )
        batch_single = scale_batch_fields(batch_single, inplace=False)

        depth_pred = preds_single.get(PredictionField.DEPTHS, None)
        depth_gt = batch_single.get(DataField.DEPTHS, None)
        images = batch_single.get(DataField.IMAGES, None)
        point_masks = batch_single.get(DataField.POINT_MASKS, None)

        if depth_pred is None or depth_gt is None or images is None:
            return {}

        s = min(self.frame_idx, depth_pred.shape[0] - 1)
        img_np = images[s].detach().float().permute(1, 2, 0).cpu().numpy()
        dp = depth_pred[s].detach().float()  # (H, W) tensor
        dg = depth_gt[s].detach().float()  # (H, W) tensor
        mask_t = point_masks[s].detach().bool() if point_masks is not None else (dg > 0)
        mask_t = mask_t & (dg > 0)

        dp = align_median(dp, dg, mask_t)

        mask = mask_t.cpu().numpy()
        dp = dp.cpu().numpy()
        dg = dg.cpu().numpy()

        pil = _render_depth_comparison(img_np, dp, dg, mask)

        # Save to disk
        seq_name = batch_single.get(DataField.SEQ_NAME, f"batch_{batch_idx:05d}")
        os.makedirs(output_dir, exist_ok=True)
        pil.save(os.path.join(output_dir, f"{seq_name}_depth_vis.png"))
        if trackers and global_step is not None:
            wandb_tracker = next((t for t in trackers if t.name == "wandb"), None)
            if wandb_tracker is not None:
                wandb_tracker.log(
                    {f"val/{dataset_name}/depth_vis": wandb.Image(pil, caption=seq_name)},
                    step=global_step,
                )

        return {}
