# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
"""Dataverse dataset wrappers for training and evaluation.

The goal is to keep *one* authoritative implementation of Dataverse‐specific
logic (field lists, etc.) while avoiding multiple
inheritance problems.  We achieve this by:

1.  Collecting all shared helpers as *functions* in this module.
2.  Implementing two thin classes – `DataverseTrainDataset` and
    `DataverseEvalDataset` – that inherit **only** from the framework-specific
    base they need (`TrainDataset` or `EvalDataset`).

Both classes call the functional helpers, so there is still exactly one place
where the tricky Dataverse details live, but no class hierarchy conflicts.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from dvlt.common.constants import DataField
from dvlt.data.sources import dataset_from_config
from dvlt.data.sources.datasets import DataField as DV, DataSource
from dvlt.common.numpy.geometry import depth_to_world_coords_points
from dvlt.data.datasets.eval import EvalDataset
from dvlt.data.datasets.train import TrainDataset
from dvlt.data.datasets.util import sample_scale_aug_factor


DEFAULT_FIELDS: List[DV] = [
    DV.IMAGE_RGB,
    DV.CAMERA_C2W_TRANSFORM,
    DV.CAMERA_INTRINSICS,
    DV.CAMERA_MODEL,
    DV.DEPTH,
]


def _build_K(raw: np.ndarray) -> np.ndarray:  # raw = [fx, fy, cx, cy]
    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[2, 2] = raw[0], raw[1], raw[2], raw[3], 1.0
    return K


def read_frames(
    ds,
    video_id: int,
    frame_idxs: List[int],
    *,
    view_idxs: Optional[List[int]] = None,
    requested_datafields: Optional[List[str]] = None,
    background_masks: Optional[bool] = False,
):
    """Read frames from a dataverse dataset and return plain numpy lists.

    If requested_datafields is provided (list of DataField constants), only the
    corresponding minimal set of Dataverse fields will be read. Otherwise,
    a minimal DEFAULT_FIELDS baseline is read.
    """

    # Choose which DV fields to read strictly from requested higher-level fields
    if requested_datafields is None:
        # Minimal baseline if not provided
        fields = DEFAULT_FIELDS.copy()
    else:
        fields = _datafields_to_dv_fields(requested_datafields)
        if len(fields) == 0:
            fields = DEFAULT_FIELDS.copy()

    if background_masks:
        fields.append(DV.BACKGROUND_MASK)

    available_fields = ds.available_data_fields()
    fields = [f for f in fields if f in available_fields]
    data = ds.read(video_id, frame_idxs=frame_idxs, view_idxs=view_idxs, data_fields=fields)
    seq_name = data.get(DV.KEY, f"{video_id:06d}")

    # IMAGE_RGB can be a stacked tensor [S, 3, H, W] or a list of [3, H, W]
    rgb = data.get(DV.IMAGE_RGB)
    if rgb is not None:
        if not isinstance(rgb, (list, tuple)):
            rgb = [im for im in rgb]
        images = [(im.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for im in rgb]
    else:
        images = None

    if DV.DEPTH in data:
        depth_val = data[DV.DEPTH]
        if not isinstance(depth_val, (list, tuple)):
            depth_val = depth_val.numpy()
        depths = [d.numpy() if isinstance(d, torch.Tensor) else d for d in depth_val]
    else:
        depths = None

    Ks = [_build_K(raw) for raw in data[DV.CAMERA_INTRINSICS].cpu().numpy()] if DV.CAMERA_INTRINSICS in data else None
    exts_arr = data[DV.CAMERA_C2W_TRANSFORM].cpu().numpy() if DV.CAMERA_C2W_TRANSFORM in data else None
    exts = [e for e in exts_arr] if exts_arr is not None else None

    # background masks
    if DV.BACKGROUND_MASK in data:
        background_masks = data[DV.BACKGROUND_MASK].cpu().numpy().astype(bool)
    else:
        background_masks = None

    return (
        seq_name,
        images,
        depths,
        Ks,
        exts,
        background_masks,
    )


def _datafields_to_dv_fields(requested: List[str]) -> List[DV]:
    # Declarative base mapping from higher-level DataField -> required DV fields
    base_requirements: dict[str, set[DV]] = {
        DataField.IMAGES: {DV.IMAGE_RGB},
        DataField.EXTRINSICS_C2W: {DV.CAMERA_C2W_TRANSFORM},
        DataField.INTRINSICS: {DV.CAMERA_INTRINSICS},
        DataField.DEPTHS: {DV.DEPTH},
        # Derived geometry
        DataField.WORLD_POINTS: {DV.DEPTH, DV.CAMERA_C2W_TRANSFORM, DV.CAMERA_INTRINSICS},
        DataField.POINT_MASKS: {DV.DEPTH, DV.CAMERA_C2W_TRANSFORM, DV.CAMERA_INTRINSICS},
    }

    requested_set = set(requested)
    required: set[DV] = set()

    # Accumulate static requirements
    for df, dv_fields in base_requirements.items():
        if df in requested_set:
            required.update(dv_fields)

    return list(required)


def _dv_fields_to_datafields(fields: List[DV]) -> List[str]:
    # Declarative mapping for direct one-to-one DV -> DataField
    direct_map: dict[DV, str] = {
        DV.IMAGE_RGB: DataField.IMAGES,
        DV.CAMERA_C2W_TRANSFORM: DataField.EXTRINSICS_C2W,
        DV.CAMERA_INTRINSICS: DataField.INTRINSICS,
        DV.DEPTH: DataField.DEPTHS,
    }

    fields_set = set(fields)
    result: list[str] = []

    # Apply direct mappings first
    for dv_field, df in direct_map.items():
        if dv_field in fields_set:
            result.append(df)

    # Multi-field derivations
    if {
        DV.DEPTH,
        DV.CAMERA_C2W_TRANSFORM,
        DV.CAMERA_INTRINSICS,
    }.issubset(fields_set):
        result.append(DataField.WORLD_POINTS)
        result.append(DataField.POINT_MASKS)

    return result


def _compute_requested_datafields(data_fields: Optional[List[str]]) -> Optional[list[str]]:
    """Return requested fields as-is. Baseline defaults are assigned in higher-level wrappers."""
    return list(data_fields) if data_fields is not None else None


def _filter_sample_to_requested(sample: dict, requested_fields: Optional[List[str]]) -> dict:
    if requested_fields is None:
        return sample
    # Metadata keys should never be filtered out
    # Sequence metadata should never be filtered out
    metadata_keys = {
        DataField.SEQ_NAME,
        DataField.IDS,
        DataField.ORIGINAL_SIZES,
        DataField.SCALE_FACTOR,
        DataField.CONSISTENT_AUG,
        DataField.METRIC_SCALE,
        DataField.IS_SYNTHETIC,
    }
    keep_keys = metadata_keys.union(set(requested_fields))
    return {k: v for k, v in sample.items() if (k in keep_keys and v is not None)}


class DataverseTrainDataset(TrainDataset):
    """Dataverse variant that produces training samples."""

    def __init__(
        self,
        *args,
        view_indices: Optional[List[int]] = None,
        dataverse_cfg: str | DictConfig,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cfg = OmegaConf.load(dataverse_cfg) if isinstance(dataverse_cfg, str) else dataverse_cfg
        self.view_indices = view_indices
        self._ds = None

    @property
    def ds(self):
        """Lazy-load the dataset from config."""
        if self._ds is None:
            self._ds = dataset_from_config(self.cfg)
        return self._ds

    def __len__(self):
        return self.ds.num_videos()

    def num_frames(self, seq_idx: int) -> int:
        """Return number of frames for a specific sequence."""
        return self.ds.num_frames(seq_idx)

    def num_views(self, seq_idx: int) -> int:
        """Return number of views for a specific sequence."""
        if self.view_indices is not None:
            return len(self.view_indices)
        return self.ds.num_views(seq_idx)

    def get_metadata(self, seq_idx: int):
        if hasattr(self.ds, "get_dataset_metadata"):
            return self.ds.get_dataset_metadata()
        else:
            return None

    def available_data_fields(self):
        return _dv_fields_to_datafields(self.ds.available_data_fields())

    def get_data(
        self,
        seq_index: int,
        data_fields: Optional[List[str]] = None,
        img_per_seq: Optional[int] = None,
        aspect_ratio: float = 1.0,
    ):
        assert img_per_seq is not None
        video_len = self.ds.num_frames(seq_index)

        if self.view_indices is not None:
            view_idx = random.choice(self.view_indices)
            assert view_idx < self.ds.num_views(
                seq_index
            ), f"View index {view_idx} is out of range for video {seq_index}"
        else:
            num_views = self.ds.num_views(seq_index)
            view_idx = random.randint(0, num_views - 1)

        metadata = self.get_metadata(seq_index)
        is_synthetic = metadata is not None and metadata.data_origin == DataSource.SYNTHETIC
        metric_scale = metadata is not None and metadata.is_metric_scale

        if img_per_seq == 1:
            indices = np.array([random.randint(0, video_len - 1)])
        else:
            # pose array for nearby-frame sampling
            all_c2w = self.ds.read(
                seq_index,
                frame_idxs=list(range(video_len)),
                view_idxs=[view_idx] * video_len,
                data_fields=[DV.CAMERA_C2W_TRANSFORM],
            )[DV.CAMERA_C2W_TRANSFORM]
            indices = self.pick_neighbor_views(random.randint(0, video_len - 1), img_per_seq, video_len, all_c2w)

        # Determine which higher-level fields are requested
        requested_datafields = _compute_requested_datafields(data_fields)

        (
            seq_name,
            images,
            depths,
            Ks,
            exts,
            background_masks,
        ) = read_frames(
            self.ds,
            seq_index,
            indices,
            view_idxs=[view_idx] * len(indices),
            requested_datafields=requested_datafields,
            background_masks=self.depth_filter_background,
        )

        target_hw = self.get_target_hw(aspect_ratio)
        imgs_out: list[np.ndarray] = []
        depths_out: list[np.ndarray] = []
        wp_out: list[np.ndarray] = []
        masks_out: list[np.ndarray] = []
        exts_out: list[np.ndarray] = []
        Ks_out: list[np.ndarray] = []
        orig_sz: list[Tuple[int, int]] = []

        # Decide whether augmentations should be consistent across frames
        consistent_aug = random.random() <= self.consistent_aug_ratio

        # Sample scale augmentation factor: once for all frames if consistent, else per-frame
        scale_aug_factor = sample_scale_aug_factor(self.scale_aug_max) if (consistent_aug and self.scale_aug) else None

        for i, (img, depth, ext, K) in enumerate(zip(images, depths, exts, Ks)):
            background_mask_i = background_masks[i] if background_masks is not None else None

            (
                img_p,
                depth_p,
                ext_p,
                K_p,
                wp,
                _,
                mask,
            ) = self.prepare_view(
                image=img,
                depth_map=depth,
                extrinsics_c2w=ext,
                intrinsics=K,
                original_hw=img.shape[:2],
                target_hw=target_hw,
                background_mask=background_mask_i,
                filepath=f"{self.name}_{seq_name}_{i}.png",
                scale_aug_factor=scale_aug_factor,
            )

            if not consistent_aug and self.scale_aug:
                scale_aug_factor = sample_scale_aug_factor(self.scale_aug_max)

            imgs_out.append(img_p)
            depths_out.append(depth_p)
            exts_out.append(ext_p)
            Ks_out.append(K_p)
            wp_out.append(wp)
            masks_out.append(mask)
            orig_sz.append(img.shape[:2])

        indices_np = np.asarray(indices, dtype=np.int64)
        sample = {
            DataField.SEQ_NAME: f"{self.name}_{seq_name}",
            DataField.IS_SYNTHETIC: is_synthetic,
            DataField.METRIC_SCALE: metric_scale,
            DataField.IDS: indices_np,
            DataField.IMAGES: imgs_out,
            DataField.DEPTHS: depths_out,
            DataField.EXTRINSICS_C2W: exts_out,
            DataField.INTRINSICS: Ks_out,
            DataField.WORLD_POINTS: wp_out,
            DataField.POINT_MASKS: masks_out,
            DataField.ORIGINAL_SIZES: orig_sz,
            DataField.CONSISTENT_AUG: consistent_aug,
        }

        return _filter_sample_to_requested(sample, data_fields)


class DataverseEvalDataset(EvalDataset):
    """Evaluation-time Dataverse wrapper (deterministic subsampling)."""

    def __init__(
        self,
        *args,
        dataverse_cfg: str | DictConfig,
        video_id: Optional[int | str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cfg = OmegaConf.load(dataverse_cfg) if isinstance(dataverse_cfg, str) else dataverse_cfg
        self._ds = None
        self._video_id_input = video_id
        self._resolved_video_id = None

    @property
    def ds(self):
        """Lazy-load the dataset from config."""
        if self._ds is None:
            self._ds = dataset_from_config(self.cfg)
        return self._ds

    def _get_resolved_video_id(self):
        if self._resolved_video_id is None:
            if isinstance(self._video_id_input, str):
                self._resolved_video_id = self.ds.get_video_id_by_name(self._video_id_input)
            else:
                self._resolved_video_id = self._video_id_input
        return self._resolved_video_id

    def __len__(self):
        return self.ds.num_videos() if self._video_id_input is None else 1

    def available_data_fields(self):
        return _dv_fields_to_datafields(self.ds.available_data_fields())

    def get_data(self, seq_index: int, data_fields: Optional[List[str]] = None):
        video_id = self._get_resolved_video_id() if self._video_id_input is not None else seq_index
        video_len = self.ds.num_frames(video_id)

        if DV.CAMERA_C2W_TRANSFORM in self.ds.available_data_fields():
            # pose array for nearby-frame sampling
            all_c2w = self.ds.read(
                video_id,
                frame_idxs=list(range(video_len)),
                view_idxs=[0]
                * video_len,  # TODO how to best test multiple view indices? Treat them as different videos?
                data_fields=[DV.CAMERA_C2W_TRANSFORM],
            )[DV.CAMERA_C2W_TRANSFORM]
        else:
            all_c2w = None

        idxs = self._get_indices(video_len, self.max_frames, all_c2w)

        # Determine which higher-level fields are requested
        requested_datafields = _compute_requested_datafields(data_fields)

        (
            seq_name,
            images,
            depths,
            Ks,
            exts,
            _,  # background_masks (not used for eval right now)
        ) = read_frames(
            self.ds,
            video_id,
            idxs,
            requested_datafields=requested_datafields,
        )

        (
            imgs_proc,
            depths_proc,
            Ks_proc,
        ) = self._preprocess_images_depths_intrinsics(
            images,
            depths,
            Ks,
        )

        wp_out, mask_out = [], []
        if depths_proc is not None and exts is not None and Ks_proc is not None:
            for d, ext, K in zip(depths_proc, exts, Ks_proc):
                wp, _, pm = depth_to_world_coords_points(d, ext, K)
                wp_out.append(wp)
                mask_out.append(pm)
        else:
            wp_out = None
            mask_out = None

        idxs_np = np.asarray(idxs, dtype=np.int64)
        sample = {
            DataField.SEQ_NAME: f"{self.name}_{seq_name}",
            DataField.IDS: idxs_np,
            DataField.IMAGES: imgs_proc,
            DataField.EXTRINSICS_C2W: exts,
            DataField.INTRINSICS: Ks_proc,
            DataField.DEPTHS: depths_proc,
            DataField.WORLD_POINTS: wp_out,
            DataField.POINT_MASKS: mask_out,
            DataField.ORIGINAL_SIZES: [(im.shape[0], im.shape[1]) for im in images],
        }

        return _filter_sample_to_requested(sample, data_fields)
