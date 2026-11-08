# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Base class for *training* datasets in dvlt.

Concrete dataset parsers (Co3D, ScanNet++, Kubric, ...) inherit from
:class:`TrainDataset` and implement :meth:`get_data`. The base class provides
the shared "given a frame, here is the canonical preprocessing pipeline"
logic in :meth:`prepare_view`, plus a small helper for picking a
window of frames around a given anchor in :meth:`pick_neighbor_views`.

This file is dvlt-authored. The high-level pipeline (principal-point crop ->
landscape/portrait check -> resize with overshoot -> strict re-crop ->
optional 90° rotation -> unproject to world points) is a fairly standard
recipe in multi-view dataset code; the expressed control flow, the
configuration surface, and the augmentation knobs were re-implemented for
dvlt.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dvlt.common.numpy.geometry import depth_to_world_coords_points
from dvlt.data.datasets.util import (
    clip_depth_outliers,
    crop_around_principal_point,
    rank_views_by_pose_similarity,
    resize_with_overshoot,
    rotate_frame_90,
)


# Permit very large images coming out of dataset parsers.
Image.MAX_IMAGE_PIXELS = None


# Default "view ranking" mode: pick the next ``N-1`` frames by camera-pose
# similarity (good for unordered image collections like ETH3D 360°).
_RANKING_POSE = "pose"
# Default "view ranking" mode: pick the next ``N-1`` frames by their index
# in the underlying sequence (good for video data with temporal continuity).
_RANKING_INDEX = "index"
# View-sampling modes.
_SAMPLING_RANDOM = "random"
_SAMPLING_ORDERED = "random_ordered"

# Default window size around the anchor frame when no window_size /
# window_size_ratio is explicitly configured. Larger values make it easier for the
# loader to find a valid frame triplet at the cost of weaker pose-distance
# inductive bias.
_DEFAULT_WINDOW_SIZE = 256


class TrainDataset(Dataset):
    """Common scaffolding for dvlt training datasets.

    Subclasses must implement :meth:`get_data`, :meth:`available_data_fields`,
    :meth:`num_frames`, and :meth:`num_views`. The base class:

    * Routes ``__getitem__`` (both ``int`` and 3/4-tuple forms) through
      ``get_data``.
    * Provides :meth:`prepare_view` for the standard "crop / resize /
      rotate / unproject" pipeline.
    * Provides :meth:`pick_neighbor_views` for picking an anchor + ``N-1``
      nearby frames either by index-window or by pose-similarity ranking.
    """

    def __init__(
        self,
        view_ranking: str = _RANKING_POSE,
        view_sampling: str = _SAMPLING_RANDOM,
        depth_filter_background: bool = False,
        depth_filter_percentile: int = -1,
        max_depth_thresh: Optional[float] = None,
        rescale: bool = True,
        scale_aug: bool = True,
        scale_aug_max: float = 0.1,
        allow_orientation_swap: bool = True,
        window_size_ratio: Optional[float] = None,
        window_size: Optional[int] = None,
        consistent_aug_ratio: float = 0.7,
    ):
        """Configure the dataset behaviour.

        Args:
            view_ranking: one of ``"pose"`` (rank candidate frames by SO(3)+t
                similarity, useful for unordered collections) or ``"index"``
                (use the natural sequence order, useful for video).
            view_sampling: ``"random"`` or ``"random_ordered"``. The latter
                additionally sorts the sampled ids so the model sees a
                temporally-ordered window.
            depth_filter_background: if True and a background mask is passed
                to :meth:`prepare_view`, set the masked pixels' depth to 0.
            depth_filter_percentile: top-percentile cutoff for depth outlier
                clamping. ``<= 0`` disables filtering.
            max_depth_thresh: optional absolute upper bound for valid depth.
            rescale: when False, skip the overshoot-resize step and rely on
                the input image already being close to the target size.
            scale_aug: enable random zoom-in augmentation during ``prepare_view``.
            scale_aug_max: max value of the triangular zoom-in factor (mode 0).
            allow_orientation_swap: when True, randomly rotate landscape inputs to
                portrait so the network sees both orientations.
            window_size_ratio: window size around the anchor expressed as a
                multiple of ``total_ids``. Ignored when ``window_size`` is
                given.
            window_size: window size around the anchor expressed as an
                absolute number of frames. Defaults to ``_DEFAULT_WINDOW_SIZE``
                when neither this nor ``window_size_ratio`` is set.
            consistent_aug_ratio: probability (in ``[0, 1]``) that geometric
                + appearance augmentations are applied identically across the
                frames of a single sequence.
        """
        super().__init__()
        self.img_size: Optional[int] = None
        self.patch_size: Optional[int] = None

        self.view_ranking = view_ranking
        self.view_sampling = view_sampling

        self.depth_filter_background = depth_filter_background
        self.depth_filter_percentile = depth_filter_percentile
        self.max_depth_thresh = max_depth_thresh

        self.rescale = rescale
        self.scale_aug = scale_aug
        self.scale_aug_max = scale_aug_max
        self.allow_orientation_swap = allow_orientation_swap

        if window_size is None and window_size_ratio is None:
            window_size = _DEFAULT_WINDOW_SIZE
        self.window_size_ratio = window_size_ratio
        self.window_size = window_size
        self.consistent_aug_ratio = consistent_aug_ratio

        self.name: str = "null"

    # ----- bookkeeping setters (called by the composing dataset wrapper) -----

    def set_image_params(self, img_size: int, patch_size: int) -> None:
        """Pin the target image side and the ViT patch size."""
        self.img_size = img_size
        self.patch_size = patch_size

    def set_name(self, name: str) -> None:
        """Record a human-readable dataset name for logging."""
        self.name = name

    # ----- abstract slots that concrete datasets must fill in -----

    def num_frames(self, seq_idx: int) -> int:
        """Return the total number of frames in sequence ``seq_idx``."""
        raise NotImplementedError

    def num_views(self, seq_idx: int) -> int:
        """Return the total number of viewpoints in sequence ``seq_idx``."""
        raise NotImplementedError

    def available_data_fields(self):
        """List the :class:`DataField` enums this dataset can populate."""
        raise NotImplementedError

    def get_data(
        self,
        seq_index: int,
        data_fields: Optional[List[str]] = None,
        img_per_seq: Optional[int] = None,
        aspect_ratio: float = 1.0,
    ) -> Dict[str, Any]:
        """Build a raw sample dict for sequence ``seq_index``.

        Implementations should return numpy arrays (one per frame, stacked or
        as a list-of-arrays); tensor conversion / validation happens upstream
        in :class:`dvlt.data.datasets.multi_source.MultiSourceDataset`.
        """
        raise NotImplementedError

    # ----- standard __getitem__ routing -----

    def __getitem__(
        self,
        idx_N: Union[int, Tuple[int, Optional[int], float, Optional[list[str]]]],
    ) -> Dict[str, Any]:
        """Route ``int`` / ``(seq, k, aspect[, fields])`` indices into :meth:`get_data`."""
        assert (
            self.img_size is not None and self.patch_size is not None
        ), "Image size and patch size must be set via set_image_params() before indexing."
        assert self.img_size % self.patch_size == 0, "img_size must be divisible by patch_size"

        # Defaults that mirror the int-index path.
        seq_index: int
        img_per_seq: Optional[int] = None
        aspect_ratio: float = 1.0
        data_fields: Optional[List[str]] = None

        if isinstance(idx_N, tuple):
            if len(idx_N) == 4:
                seq_index, img_per_seq, aspect_ratio, data_fields = idx_N
            elif len(idx_N) == 3:
                seq_index, img_per_seq, aspect_ratio = idx_N
            else:
                raise ValueError(f"Tuple index must have 3 or 4 elements, got {len(idx_N)}")
        else:
            seq_index = idx_N

        return self.get_data(
            seq_index=seq_index,
            data_fields=data_fields,
            img_per_seq=img_per_seq,
            aspect_ratio=aspect_ratio,
        )

    # ----- target image side from aspect ratio + patch alignment -----

    def get_target_hw(self, aspect_ratio: float) -> np.ndarray:
        """Pick a target ``(H, W)`` that respects the configured patch grid.

        Concretely, ``W`` is fixed at ``self.img_size`` and ``H`` is
        ``floor(img_size * aspect_ratio / patch_size) * patch_size`` so the
        ViT patch embed sees integer rows and columns.
        """
        long_side = int(self.img_size * aspect_ratio)
        if long_side % self.patch_size:
            long_side -= long_side % self.patch_size
        return np.array([long_side, self.img_size], dtype=np.int64)

    # ----- core single-frame pipeline -----

    def prepare_view(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        extrinsics_c2w: np.ndarray,
        intrinsics: np.ndarray,
        original_hw: np.ndarray,
        target_hw: np.ndarray,
        filepath: Optional[str] = None,
        background_mask: Optional[np.ndarray] = None,
        safe_bound: int = 4,
        scale_aug_factor: Optional[float] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Run the standard preprocessing pipeline on one frame.

        Steps, in order:

        1. Copy inputs to keep the caller's arrays pristine.
        2. Zero out non-finite or negative depths.
        3. Optionally zero depths under the background mask.
        4. Optionally clip depth by percentile / absolute bound.
        5. Principal-point-centered crop at the (possibly augmented) source
           size, to remove arbitrary off-center optical axes.
        6. Optional landscape -> portrait swap (random 90° rotation).
        7. Resize with overshoot to target_hw + safe_bound.
        8. Final strict principal-point crop to exactly target_hw.
        9. Optional 90° rotation when (6) flagged it.
        10. Depth -> world / camera coords + valid mask.

        Returns the processed tuple
        ``(image, depth, c2w[4x4], K, world_xyz, cam_xyz, point_mask)``.
        """
        # Step 1: defensive copies.
        image = np.copy(image)
        depth_map = np.copy(depth_map) if depth_map is not None else None
        extrinsics_c2w = np.copy(extrinsics_c2w)
        intrinsics = np.copy(intrinsics)

        # Step 2/3/4: depth sanitization.
        if depth_map is not None:
            img_h, img_w = image.shape[:2]
            assert depth_map.shape[:2] == (
                img_h,
                img_w,
            ), f"Depth map shape {depth_map.shape[:2]} must match image {(img_h, img_w)}"
            depth_map[~np.isfinite(depth_map) | (depth_map < 0.0)] = 0.0
            if self.depth_filter_background and background_mask is not None:
                depth_map[background_mask] = 0.0
            wants_thresholding = self.max_depth_thresh is not None or self.depth_filter_percentile > 0
            if wants_thresholding:
                depth_map = clip_depth_outliers(
                    depth_map,
                    max_percentile=self.depth_filter_percentile,
                    max_depth=self.max_depth_thresh,
                )

        # Step 5: principal-point centered crop at the source size.
        image, depth_map, intrinsics = crop_around_principal_point(
            image, depth_map, intrinsics, original_hw, filepath=filepath
        )

        # Step 6: landscape -> portrait orientation swap.
        current_hw = np.array(image.shape[:2])
        effective_target_hw = target_hw
        rotate_to_landscape = False
        if self.allow_orientation_swap and current_hw[0] > 1.2 * current_hw[1]:
            if (target_hw[0] != target_hw[1]) and (np.random.rand() > 0.5):
                effective_target_hw = np.array([target_hw[1], target_hw[0]])
                rotate_to_landscape = True

        # Step 7: resize with overshoot.
        if self.rescale:
            image, depth_map, intrinsics = resize_with_overshoot(
                image,
                depth_map,
                intrinsics,
                effective_target_hw,
                current_hw,
                safe_bound=safe_bound,
                scale_aug_factor=scale_aug_factor,
            )

        # Step 8: final strict crop to the exact target_hw (possibly swapped).
        image, depth_map, intrinsics = crop_around_principal_point(
            image, depth_map, intrinsics, effective_target_hw, filepath=filepath, strict=True
        )

        # Step 9: rotate iff we asked to.
        if rotate_to_landscape:
            clockwise = bool(np.random.rand() > 0.5)
            image, depth_map, extrinsics_c2w, intrinsics = rotate_frame_90(
                image, depth_map, extrinsics_c2w, intrinsics, clockwise=clockwise
            )

        # Step 10: unproject depth -> world / camera.
        world_xyz, cam_xyz, mask = depth_to_world_coords_points(depth_map, extrinsics_c2w, intrinsics)

        # Always return a 4x4 extrinsic for downstream consistency.
        if extrinsics_c2w.shape[0] == 3:
            extrinsics_c2w = np.vstack((extrinsics_c2w, np.array([0, 0, 0, 1], dtype=extrinsics_c2w.dtype)))

        return image, depth_map, extrinsics_c2w, intrinsics, world_xyz, cam_xyz, mask

    # ----- pick anchor + ``N-1`` nearby frames -----

    def pick_neighbor_views(
        self,
        start_idx: int,
        total_ids: int,
        full_seq_num: int,
        extrinsics_c2w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample ``total_ids`` frame indices around the anchor ``start_idx``.

        The behaviour depends on :attr:`view_ranking`:

        * ``"pose"``: rank *all* frames by SO(3)+t similarity to ``start_idx``,
          then sample ``total_ids - 1`` from the top ``2 * window_size``.
          Requires ``extrinsics_c2w`` to be passed.
        * ``"index"``: sample ``total_ids - 1`` frames from the window
          ``[start_idx - window_size, start_idx + window_size]`` intersected
          with ``[0, full_seq_num)``.

        After sampling, :attr:`view_sampling` decides the order:

        * ``"random"``: keep sampled order.
        * ``"random_ordered"``: sort ascending so the network sees a
          temporally-ordered window.

        Args:
            start_idx: anchor frame index (always present in the output).
            total_ids: total number of frames to return (including the anchor).
            full_seq_num: total frames in the underlying sequence.
            extrinsics_c2w: ``(full_seq_num, 4, 4)`` poses; required for ``"pose"``
                ranking.

        Returns:
            ``(total_ids,)`` array of frame indices.
        """
        # Window size: either an absolute (window_size) or a multiple of
        # the number of frames we're sampling.
        if self.window_size is None:
            window = int(total_ids * self.window_size_ratio)
        else:
            window = self.window_size

        # Build the *valid range* of candidate indices and an optional
        # permutation that maps from "rank order" to "frame index".
        rank_order: Optional[np.ndarray] = None
        if self.view_ranking == _RANKING_POSE:
            assert extrinsics_c2w is not None, "view_ranking='pose' requires extrinsics_c2w"
            rank_order = rank_views_by_pose_similarity(start_idx, extrinsics_c2w)
            lo, hi = 0, min(full_seq_num, 2 * window)
        elif self.view_ranking == _RANKING_INDEX:
            lo = max(0, start_idx - window)
            hi = min(full_seq_num, start_idx + window)
        else:
            raise ValueError(f"Unknown view_ranking: {self.view_ranking!r}")

        candidate_idxs = np.arange(lo, hi)
        # Sample with replacement is intentional: duplicates are tolerable
        # and avoid edge cases where the window is too tight.
        sampled = np.random.choice(candidate_idxs, size=(total_ids - 1,), replace=True)
        if rank_order is not None:
            sampled = rank_order[sampled]

        ids = np.insert(sampled, 0, start_idx)

        if self.view_sampling == _SAMPLING_ORDERED:
            ids = np.sort(ids)
        elif self.view_sampling != _SAMPLING_RANDOM:
            raise ValueError(f"Unknown view_sampling: {self.view_sampling!r}")

        return ids
