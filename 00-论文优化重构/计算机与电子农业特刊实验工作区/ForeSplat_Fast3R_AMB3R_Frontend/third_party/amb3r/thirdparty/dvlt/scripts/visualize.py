# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model inference and visualization script."""

import logging
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from dvlt.callbacks.util import align_predictions_to_gt, index_batch_and_predictions, scale_batch_fields
from dvlt.callbacks.visualization import SceneVisualizationCallback, _exclude_top_l2_error_indices
from dvlt.common.constants import DataField, PredictionField
from dvlt.config.cli import cli
from dvlt.config.schema import Config, register_configs
from dvlt.engine.trainer import get_checkpoint_dir
from dvlt.struct.util import extri_intri_to_cameras
from dvlt.viz.scene_rerun import visualize_scene


@dataclass
class VisualizeModelConfig(Config):
    """Configuration for model visualization."""

    max_points: int = 1_000_000
    conf_threshold: float = 25.0
    use_point_mask: bool = True
    log_images: bool = True
    save_path: Optional[str] = None

    test_dataset_name: Optional[str] = None
    mode: str = "test"  # "train" or "test"
    batch_idx: Optional[int] = None
    coordinate_convention: str = "RDF"
    # Default to offline mode: write ``.rrd`` files under
    # ``<trainer.output_dir>/<trainer.experiment_name>/viz`` instead of
    # connecting to a live Rerun viewer. Set ``server_address`` to a
    # ``host:port`` (or full ``rerun+http://.../proxy`` URL) to stream live.
    server_address: Optional[str] = None
    image_max_size: int = 0  # if > 0, resize logged images so max(H, W) <= this; 0 = no resize


def _register_configs():
    """Register structured configs with Hydra's config store."""
    register_configs()

    cs = ConfigStore.instance()
    cs.store(name="viz_model_base", node=VisualizeModelConfig)


def visualize_predictions_rerun(
    log_name: str,
    predictions: dict,
    batch: dict,
    max_points: int = 1_000_000,
    conf_threshold: float = 25.0,
    use_point_mask: bool = True,
    log_images: bool = True,
    save_path: Optional[str] = None,
    coordinate_convention: str = "RDF",
    server_address: Optional[str] = "0.0.0.0:9876",
    image_max_size: int = 0,
    app_id: Optional[str] = None,
):
    """Visualize model predictions aligned to GT using Rerun.

    Applies the same alignment pipeline used at test time:
    index -> scale GT -> Sim3 align predictions to GT.
    Both predicted and GT point clouds / cameras are logged on separate layers.

    When GT ``point_masks`` are present they are used to filter the point cloud.
    Otherwise, confidence-based percentile filtering is applied (keeps points
    above the ``conf_threshold`` percentile).

    Args:
        log_name: Log identifier for Rerun.
        predictions: Model predictions dictionary with PredictionField keys (batched).
        batch: Input batch containing ground truth data (batched).
        max_points: Maximum number of points to visualize per split.
            Set to ``0`` or ``-1`` to disable downsampling entirely.
        conf_threshold: Confidence percentile threshold used when GT point masks
            are unavailable or disabled. Points below this percentile are discarded.
        use_point_mask: Whether to use GT point masks for filtering when available.
            If False, confidence-based filtering is used regardless.
        log_images: Whether to log images to visualization.
        save_path: Optional path to save .rrd visualization.
        coordinate_convention: View coordinate convention (default "RDF").
        server_address: Rerun TCP server address.
        image_max_size: If > 0, logged images are resized so max(H, W) <= image_max_size.
            File size scales as area, so 256 is roughly 4x smaller than 512. 0 = no resize.
        app_id: Optional Rerun ``application_id``. If unset, ``log_name`` is used (default
            Rerun behavior). Pass a method-qualified value (e.g. ``"da3-G/dtu_scan1"``) when
            serving .rrd files from multiple methods so each gets its own entry in the
            recordings sidebar.
    """
    batch, predictions = index_batch_and_predictions(batch, predictions, batch_idx=0, seq_idxs=None, inplace=False)
    batch = scale_batch_fields(batch, inplace=False)
    try:
        predictions = align_predictions_to_gt(batch, predictions, inplace=False)
    except (ZeroDivisionError, RuntimeError):
        print(f"[{log_name}] WARNING: Sim3 alignment failed (no valid GT points?), showing unaligned predictions")

    images = batch[DataField.IMAGES]
    images_np = images.cpu().permute(0, 2, 3, 1).numpy()  # (S, H, W, 3)
    world_gt = batch.get(DataField.WORLD_POINTS, None)
    point_mask = batch.get(DataField.POINT_MASKS, None)
    extrinsics_c2w_gt = batch.get(DataField.EXTRINSICS_C2W, None)
    intrinsics_gt = batch.get(DataField.INTRINSICS, None)

    cameras_pred = predictions[PredictionField.CAMERAS]
    world_pred = predictions[PredictionField.WORLD_POINTS]

    # max_points <= 0 means no downsampling
    effective_max = max_points if max_points > 0 else int(1e12)

    # Build GT cameras
    cameras_gt = None
    if extrinsics_c2w_gt is not None and intrinsics_gt is not None:
        cameras_gt = extri_intri_to_cameras(extrinsics_c2w_gt, intrinsics_gt, images.shape[-2:])

    if point_mask is not None and use_point_mask:
        # GT masks available — use the standard path
        pts_pred, pts_gt, pred_rgb, gt_rgb = SceneVisualizationCallback._prepare_pointcloud_data(
            world_pred, images, world_gt, point_mask, max_points=effective_max
        )
    else:
        # No GT masks — fall back to confidence-based filtering
        confidence = predictions.get(
            PredictionField.WORLD_POINTS_DIRECT_CONF,
            predictions.get(PredictionField.DEPTHS_CONF, None),
        )

        pts_np = world_pred.detach().cpu().numpy().reshape(-1, 3)
        colors_np = (images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).reshape(-1, 3)
        gt_np = world_gt.detach().cpu().numpy().reshape(-1, 3) if world_gt is not None else None

        if confidence is not None:
            conf_flat = confidence.detach().cpu().numpy().reshape(-1)
            threshold_val = np.percentile(conf_flat, conf_threshold)
            valid = conf_flat >= threshold_val
            pts_np = pts_np[valid]
            colors_np = colors_np[valid]
            if gt_np is not None:
                gt_np = gt_np[valid]

        if len(pts_np) > effective_max:
            if gt_np is not None and len(gt_np) == len(pts_np):
                idx_keep = _exclude_top_l2_error_indices(pts_np, gt_np, np.arange(len(pts_np)))
                pts_np = pts_np[idx_keep]
                colors_np = colors_np[idx_keep]
            if len(pts_np) > effective_max:
                idx = np.random.choice(len(pts_np), effective_max, replace=False)
                pts_np = pts_np[idx]
                colors_np = colors_np[idx]

        pts_pred, pred_rgb = pts_np, colors_np.astype(np.uint8)

        # Still log GT pointcloud using the GT mask when available
        if world_gt is not None and point_mask is not None:
            gt_mask = point_mask.detach().cpu().numpy().astype(bool).reshape(-1)
            gt_all = world_gt.detach().cpu().numpy().reshape(-1, 3)
            gt_colors_all = (images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).reshape(-1, 3)
            pts_gt = gt_all[gt_mask]
            gt_rgb = gt_colors_all[gt_mask].astype(np.uint8)
            if len(pts_gt) > effective_max:
                idx = np.random.choice(len(pts_gt), effective_max, replace=False)
                pts_gt = pts_gt[idx]
                gt_rgb = gt_rgb[idx]
        else:
            pts_gt, gt_rgb = None, None

    # Print per-point L2 distance stats between aligned pred and GT
    if pts_gt is not None and pts_gt.size > 0 and pts_pred.shape[0] == pts_gt.shape[0]:
        l2 = np.linalg.norm(pts_pred.reshape(-1, 3) - pts_gt.reshape(-1, 3), axis=-1)
        print(
            f"[{log_name}] L2 point error (after Sim3 alignment) — "
            f"avg: {l2.mean():.4f}, min: {l2.min():.4f}, max: {l2.max():.4f}, "
            f"median: {np.median(l2):.4f}, n_points: {len(l2)}"
        )

    # Build camera dict — prefix with "cameras/" so all cameras sit under one
    # toggleable group in the Rerun entity tree.
    cameras_dict = {"cameras/pred": cameras_pred}
    if cameras_gt is not None:
        cameras_dict["cameras/gt"] = cameras_gt

    # Build points/rgb dicts — prefix with "pointcloud/" so pred and gt sit
    # under one toggleable group in the Rerun entity tree.
    points_dict = {"pointcloud/pred": pts_pred}
    rgb_dict = {"pointcloud/pred": pred_rgb}
    if pts_gt is not None and pts_gt.size > 0:
        points_dict["pointcloud/gt"] = pts_gt
        rgb_dict["pointcloud/gt"] = gt_rgb

    # Build images dict (keyed to match camera dict)
    images_dict = {"cameras/pred": images_np} if log_images else None

    visualize_scene(
        log_path=log_name,
        server_address=server_address,
        cameras=cameras_dict,
        points=points_dict,
        rgb=rgb_dict,
        images=images_dict,
        save_path=save_path,
        view_coordinates=coordinate_convention,
        image_max_size=image_max_size,
        max_num_points=effective_max,
        app_id=app_id,
    )


_register_configs()


@cli(config_path="../config/experiments", config_name="viz_model_config", version_base=None)
def main(config: VisualizeModelConfig):
    """Main function for model visualization."""
    config = VisualizeModelConfig(**config)
    # Setup accelerator
    accelerator = Accelerator(mixed_precision=config.trainer.mixed_precision)
    logger = get_logger(__name__)
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(f"Using device: {accelerator.device}")
    logger.info(f"Mixed precision: {config.trainer.mixed_precision}")

    # Initialize model
    logger.info("Initializing model...")
    model = instantiate(config.model)

    # Load checkpoint if available
    if config.trainer.timestamp is not None:
        output_dir = Path(config.trainer.output_dir, config.trainer.experiment_name, config.trainer.timestamp)
        ckpt = get_checkpoint_dir(output_dir)
    else:
        ckpt = config.trainer.ckpt_dir

    if ckpt == "":
        logger.warning("No checkpoint found in output_dir and ckpt_dir is empty. Skipping checkpoint loading.")
    else:
        logger.info(f"Loading checkpoint from {ckpt}")
        model.load_pretrained(ckpt, strict=True)

    # Setup model for testing
    model.setup_test(accelerator)

    # Instantiate data module
    logger.info("Setting up data...")
    data = instantiate(config.data)
    # Patch distributed_eval to True to prepare dataloader with acelerate (automatic device placement)
    data.distributed_eval = True

    # Build a list of (dataset_name_or_None, dataloader) pairs to process sequentially.
    # In test mode with no test_dataset_name, iterate over every test dataset (each saved
    # to its own subdir under save_path). Otherwise, single dataloader, no subdir.
    logger.info(f"Getting {config.mode} dataloader(s)...")
    save_in_subdirs = False
    if config.mode == "train":
        dataloader = data.train_dataloader(accelerator)
        if getattr(config.trainer, "single_sequence_overfit", False):
            logger.info("Patching sampler for single_sequence_overfit (fixed seq_idx=0)")
            sampler = dataloader.batch_sampler
            _original_generate = sampler._generate_indices

            def _fixed_generate_indices():
                _original_generate()
                sampler.indices = [0] * len(sampler.indices)

            sampler._generate_indices = _fixed_generate_indices
            _fixed_generate_indices()
        dataloaders_to_process = [(None, dataloader)]
    elif config.mode == "test":
        all_dataloaders = data.test_dataloaders(accelerator)
        if config.test_dataset_name is not None:
            dataloaders_to_process = [(config.test_dataset_name, all_dataloaders[config.test_dataset_name])]
        else:
            dataloaders_to_process = list(all_dataloaders.items())
            save_in_subdirs = True
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

    # If neither a live server nor an explicit save path was provided, default to
    # writing .rrd files under ``<trainer.output_dir>/<experiment_name>/viz`` so
    # that running ``visualize`` is fully self-contained / offline by default.
    effective_save_path = config.save_path
    if effective_save_path is None and config.server_address is None:
        effective_save_path = str(Path(config.trainer.output_dir, config.trainer.experiment_name, "viz"))
        logger.info(f"No server_address or save_path provided; defaulting to save_path={effective_save_path}")

    # Process datasets sequentially (model loaded once)
    with torch.no_grad():
        for ds_name, dataloader in dataloaders_to_process:
            if ds_name is not None:
                logger.info(f"=== Visualizing dataset: {ds_name} ===")

            # If batch_idx is specified, get only that batch; otherwise iterate through all
            if config.batch_idx is not None:
                try:
                    batch = next(islice(dataloader, config.batch_idx, config.batch_idx + 1))
                    batches_to_process = [(config.batch_idx, batch)]
                except StopIteration:
                    raise ValueError(f"batch_idx {config.batch_idx} is out of range for the dataloader") from None
            else:
                batches_to_process = enumerate(dataloader)

            for batch_idx, batch in batches_to_process:
                seq_name = batch.get(DataField.SEQ_NAME, [f"batch_{batch_idx}"])[0].replace("/", "_")
                logger.info(
                    f"Processing batch {batch_idx} ({seq_name}) with {len(batch[DataField.IMAGES][0])} frames..."
                )

                # Run inference with automatic mixed precision
                with accelerator.autocast():
                    predictions = model.predict(batch, accelerator)

                # Setup save paths — when iterating multiple datasets, write into per-dataset subdirs
                save_path = None
                if effective_save_path is not None:
                    base = Path(effective_save_path)
                    if save_in_subdirs and ds_name is not None:
                        base = base / ds_name
                    save_path = base / f"{seq_name}.rrd"

                # Derive a method-qualified Rerun application_id so recordings from different
                # methods (e.g. da3-G, baseline-eval, ours-12steps) show as distinct entries in the
                # viewer's sidebar when loaded together. We infer the method name from the final
                # component of save_path (which the gen_rrds.sh script sets to evals/rrds/<method>).
                app_id = None
                if effective_save_path is not None:
                    method_name = Path(effective_save_path).name
                    if method_name:
                        app_id = f"{method_name}/{seq_name}"

                # Visualize (predictions are aligned to GT inside)
                visualize_predictions_rerun(
                    seq_name,
                    predictions,
                    batch=batch,
                    max_points=config.max_points,
                    conf_threshold=config.conf_threshold,
                    use_point_mask=config.use_point_mask,
                    log_images=config.log_images,
                    save_path=str(save_path) if save_path else None,
                    coordinate_convention=config.coordinate_convention,
                    server_address=config.server_address,
                    image_max_size=config.image_max_size,
                    app_id=app_id,
                )

                if save_path:
                    logger.info(f"Saved visualization to {save_path}")
                else:
                    input("Press Enter to continue...")

    accelerator.end_training()


if __name__ == "__main__":
    main()
