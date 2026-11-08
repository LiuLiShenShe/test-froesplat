# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Composed datasets: sample processing and augmentation pipelines.

Wraps one or more concrete :class:`dvlt.data.datasets.train.TrainDataset` (or
:class:`dvlt.data.datasets.eval.EvalDataset`) instances behind a single
``Dataset`` that handles concatenation, per-sample validation, tensor
conversion, optional appearance augmentation, and scene normalization.

This module is dvlt-authored. The cross-dataset bisect mapping is the
``bisect.bisect_right`` pattern from :class:`torch.utils.data.ConcatDataset`
(BSD-3-Clause). The 3-tuple ``(seq_index, img_per_seq, aspect_ratio)``
indexing protocol is part of dvlt's :class:`TrainDataset.get_data` contract
(see ``docs/data/DATA.md``). Validation, scene normalization, deterministic
per-sample seeding, length padding, and retry handling are dvlt-original.
The "consistent vs per-frame appearance augmentation" idea is broadly used
across multi-view training pipelines; this module dispatches between the
two via a per-sample ``CONSISTENT_AUG`` flag that the underlying dataset is
free to set.
"""

import bisect
import logging
import random
import traceback
from collections.abc import Mapping
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dvlt.common.constants import DataField
from dvlt.data.datasets.util import (
    build_appearance_aug,
    check_all_finite,
    compute_normalization_transform,
)


logger = logging.getLogger(__name__)


class _SampleRequest(NamedTuple):
    """Per-call payload describing what the underlying dataset should fetch."""

    seq_index: int
    img_per_seq: Optional[int]
    aspect_ratio: Optional[float]


DEFAULT_FIELDS = {
    DataField.IMAGES,
    DataField.EXTRINSICS_C2W,
    DataField.INTRINSICS,
    DataField.DEPTHS,
    DataField.WORLD_POINTS,
    DataField.POINT_MASKS,
}


class MultiSourceDataset(Dataset):
    """Unified dataset that merges dataset concatenation with train/eval preparation.

    - Concatenates multiple datasets and supports both integer and tuple indexing.
      If a tuple is provided (e.g., ``(sample_idx, ...)``), the first element selects the
      sample within the concatenated set and the full tuple is forwarded to the underlying
      dataset's retrieval method.
    - Optional ``repeat_factors`` can be provided to bias the sampling across datasets.

    Training vs Evaluation:
    - Set ``training=True`` to enable behaviours needed only for training, including:
      - data augmentations (co-jitter/per-frame jitter, grayscale, blur)
      - dataset length padding to at least ``MIN_LEN`` and retry logic up to ``MAX_RETRIES``
        to avoid returning non-finite samples
    - Evaluation leaves data unaugmented and uses the exact dataset length without padding.

    Validation and Conversion:
    - Performs lightweight runtime validation and tensor conversion for all known
      ``DataField`` entries to enforce the contract described in ``docs/data/DATA.md``.
      Checks include required-field presence, dtype/range, basic shape consistency, and
      field dependencies (e.g., point masks when extrinsics + world points are present).

    Additional Processing:
    - When extrinsics/world points are present, normalizes the scene into a canonical frame
      and rescales data consistently.

    Notes:
    - Returned samples are dictionaries keyed by ``DataField`` enums with only non-``None``
      entries populated.
    """

    MIN_LEN = 1000  # Only used when training=True
    MAX_RETRIES = 10  # Only used when training=True

    def __init__(
        self,
        datasets: Dict[str, Dataset] | Dataset,
        training: bool = False,
        # data loading related
        load_data_fields: Optional[List[str]] = None,
        repeat_factors: Dict[str, float] | None = None,
        # Augmentation related
        jitter_params: Optional[Dict[str, float]] = None,
        enable_color_jitter: bool = True,
        enable_grayscale: bool = True,
        enable_gaussian_blur: bool = True,
        # data processing related
        normalize_scene: bool = True,
    ) -> None:
        """
        Initialize the MultiSourceDataset.

        Args:
            datasets: Either (a dict of) TrainDataset(s) or EvalDataset(s).
            training: If True, enables training-time behaviors.
            load_data_fields: Optional list of fields to keep from the base dataset output.
            repeat_factors: Repeat factors per dataset when ``datasets`` is a dict.
            jitter_params: Color jittering parameters. If None, uses default values.
            enable_color_jitter: Toggle for the color-jitter step (uses ``jitter_params``).
            enable_grayscale: Toggle for the random-grayscale step.
            enable_gaussian_blur: Toggle for the random-Gaussian-blur step.
            normalize_scene: Whether to normalize the scene scale with the extrinsics and world points.
        """
        # Normalize input to a list of datasets and names (support OmegaConf DictConfig as Mapping)
        if isinstance(datasets, Mapping):
            self._datasets: list[Dataset] = list(datasets.values())
            self._names: list[str] = list(datasets.keys())
        else:
            self._datasets = [datasets]
            self._names = [type(datasets).__name__]

        # Build cumulative sizes for global indexing
        self._cumulative_sizes: list[int] = []
        total = 0
        for ds in self._datasets:
            total += len(ds)
            self._cumulative_sizes.append(total)

        # Repeat factors (only meaningful when multiple datasets)
        if repeat_factors is not None and isinstance(datasets, Mapping):
            assert set(repeat_factors.keys()) == set(self._names), (
                "All datasets must have a repeat factor: " + str(repeat_factors.keys()) + " != " + str(self._names)
            )
            self._repeat_factors = [repeat_factors[n] for n in self._names]
        else:
            self._repeat_factors = None

        # Persist config flags used by validation
        self.training = training
        self.load_data_fields = load_data_fields if load_data_fields is not None else DEFAULT_FIELDS
        self.normalize_scene = normalize_scene

        if self.load_data_fields is not None:
            for name, ds in zip(self._names, self._datasets):
                available = ds.available_data_fields()
                for req in self.load_data_fields:
                    if req not in available:
                        logger.warning(f"Requested field '{req}' not available in {name}. We will pass empty tensors.")

        # Augmentation setup – only built when *training*
        if self.training and any([enable_color_jitter, enable_grayscale, enable_gaussian_blur]):
            self.image_aug = build_appearance_aug(
                jitter_params=jitter_params if enable_color_jitter else {"p": 0.0},
                enable_grayscale=enable_grayscale,
                enable_gaussian_blur=enable_gaussian_blur,
            )
        else:
            self.image_aug = None

    @property
    def names(self) -> list[str]:
        return self._names

    @property
    def repeat_factors(self) -> Optional[List[float]]:
        return self._repeat_factors

    @property
    def cumulative_sizes(self) -> list[int]:
        return self._cumulative_sizes

    def __len__(self):
        base_len = self._cumulative_sizes[-1] if self._cumulative_sizes else 0
        if self.training:
            return max(base_len, self.MIN_LEN)
        return base_len

    def __getitem__(self, idx):
        """
        Build a processed sample from the underlying dataset output, with retries during training.
        """
        # Seed per-sample RNG for deterministic in-worker randomness (view/frame
        # selection, augmentation) across checkpoint/resume. The sample_seed is
        # generated by the sampler's sample_seed_rng which is saved/restored.
        if isinstance(idx, tuple) and len(idx) >= 4:
            sample_seed = idx[3]
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2**32))
            torch.manual_seed(sample_seed)
            idx = idx[:3]  # strip seed before downstream processing

        base_len = self._cumulative_sizes[-1] if self._cumulative_sizes else 0

        if self.training and base_len < len(self):
            if isinstance(idx, tuple) and len(idx) >= 1:
                seq_index = idx[0]
                seq_index = int(seq_index / len(self) * base_len)
                idx = (seq_index,) + idx[1:]
            elif isinstance(idx, int):
                idx = int(idx / len(self) * base_len)

        max_retry = self.MAX_RETRIES if self.training else 1
        try_idx = 0
        while True:
            dataset, request, ds_name = self._resolve_request(idx, resample=(self.training and try_idx > 0))

            try:
                get_data_kwargs: Dict[str, Any] = {
                    "seq_index": request.seq_index,
                    "data_fields": self.load_data_fields,
                }
                if self.training:
                    get_data_kwargs["img_per_seq"] = request.img_per_seq
                    get_data_kwargs["aspect_ratio"] = request.aspect_ratio
                batch = dataset.get_data(**get_data_kwargs)

                sample = self._build_sample(batch)
                if check_all_finite(sample, sample.get(DataField.SEQ_NAME, None)):
                    return sample

            except Exception as e:
                tb = traceback.format_exc()
                logger.error(
                    (
                        "Data loading error in %s (request=%s, try=%d/%d). "
                        "Continuing to retry other samples.\n"
                        "Error: %s\nStacktrace:\n%s\n"
                        "Action: Investigate the dataset sample and filesystem paths above."
                    ),
                    ds_name,
                    repr(request),
                    try_idx + 1,
                    max_retry,
                    str(e),
                    tb,
                )

            try_idx += 1
            if try_idx >= max_retry:
                raise ValueError(f"Failed to get a valid sample after {max_retry} retries")

    def _map_global_to_local(self, global_index: int) -> Tuple[int, int]:
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, global_index)
        if dataset_idx == 0:
            sample_idx = global_index
        else:
            sample_idx = global_index - self._cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def _resolve_request(self, idx, resample: bool = False) -> Tuple[Dataset, _SampleRequest, str]:
        """Resolve a global index into a (dataset, request, label) triple."""
        if isinstance(idx, tuple):
            global_index = idx[0]
            tail = idx[1:]
        elif isinstance(idx, int):
            global_index = idx
            tail = ()
        else:
            raise ValueError("Index must be int or tuple")

        ds_idx, local_seq = self._map_global_to_local(global_index)
        dataset = self._datasets[ds_idx]

        if resample:
            local_len = len(dataset)
            local_seq = int(torch.randint(low=0, high=local_len, size=(1,)).item())

        img_per_seq: Optional[int] = tail[0] if len(tail) >= 1 else None
        aspect_ratio: Optional[float] = tail[1] if len(tail) >= 2 else None

        request = _SampleRequest(
            seq_index=local_seq,
            img_per_seq=img_per_seq,
            aspect_ratio=aspect_ratio,
        )
        return dataset, request, self._names[ds_idx]

    def _validate_and_convert_field(
        self,
        data,
        *,
        required=False,
        expected_dtype=np.float32,
        tensor_dtype=None,
        stack=True,
        required_when=None,
        forbidden_when=None,
        validate_range=None,
        expected_shape_template=None,
        shape_context=None,
        allow_empty=False,
    ):
        """Validate field format and convert to tensor in one step.

        Args:
            data: The field data (can be None for optional fields)
            required: Whether this field is required (raises error if None)
            expected_dtype: Expected numpy dtype (e.g., np.float32, bool)
            tensor_dtype: Target tensor dtype (defaults to expected_dtype)
            stack: Whether to stack list of arrays (True) or convert single array (False)
            required_when: Condition - this field is required when condition is met
            forbidden_when: Condition - this field is forbidden when condition is met
            validate_range: Tuple (min, max) to validate value ranges (e.g., (0, 255) for images)
            expected_shape_template: Expected shape like "[S, H, W]" or "[N, 3]"
            shape_context: Dict with known dimensions like {"S": 2, "H": 100, "W": 100}
            allow_empty: Whether to allow empty arrays/lists

        Returns:
            torch.Tensor or None: Converted tensor, or None if data was None and not required
        """
        # Check if field is required
        if required and data is None:
            raise ValueError("Required field is missing")

        # Check required/forbidden conditions
        if required_when and data is None:
            raise ValueError("Field is required when related field is provided")

        if forbidden_when and data is not None:
            raise ValueError("Field should be None when condition is met")

        # If data is None and not required, return None
        if data is None:
            return None

        # Validate data format when present
        if expected_dtype is not None:
            if isinstance(data, list):
                # Check list elements
                for i, item in enumerate(data):
                    if not isinstance(item, np.ndarray):
                        raise ValueError(f"List element [{i}] should be numpy array, got {type(item)}")
                    if item.dtype != expected_dtype:
                        raise ValueError(
                            f"List element [{i}] should have dtype {expected_dtype.__name__}, got {item.dtype}"
                        )
                    # Validate value ranges if specified
                    if validate_range is not None:
                        min_val, max_val = validate_range
                        if item.min() < min_val or item.max() > max_val:
                            raise ValueError(
                                f"List element [{i}] values should be in range [{min_val}, {max_val}], "
                                f"got [{item.min()}, {item.max()}]"
                            )
            elif isinstance(data, np.ndarray):
                if data.dtype != expected_dtype:
                    raise ValueError(f"Should have dtype {expected_dtype.__name__}, got {data.dtype}")
                # Validate value ranges if specified
                if validate_range is not None:
                    min_val, max_val = validate_range
                    if data.min() < min_val or data.max() > max_val:
                        raise ValueError(
                            f"Values should be in range [{min_val}, {max_val}], " f"got [{data.min()}, {data.max()}]"
                        )

        # Shape validation
        if expected_shape_template and shape_context:
            self._validate_shape(data, expected_shape_template, shape_context)

        # Basic empty check (unless explicitly allowed)
        if not allow_empty:
            if isinstance(data, list) and len(data) == 0:
                raise ValueError("Should not be empty list")
            elif isinstance(data, np.ndarray) and data.size == 0:
                raise ValueError("Should not be empty array")

        # Convert to tensor
        tensor_dtype = tensor_dtype or expected_dtype
        if stack and isinstance(data, list):
            if len(data) == 0:
                # Handle empty lists - create empty tensor with appropriate shape
                if expected_shape_template and shape_context:
                    # Try to infer shape from template
                    dims = expected_shape_template.strip("[]").split(", ")
                    shape = []
                    for dim in dims:
                        if dim.isdigit():
                            shape.append(int(dim))
                        elif dim in shape_context:
                            shape.append(shape_context[dim])
                        else:
                            shape.append(0)  # Unknown dimension, use 0 for empty
                    return torch.empty(shape, dtype=torch.from_numpy(np.array([], dtype=tensor_dtype)).dtype)
                else:
                    # Fallback to simple empty tensor
                    return torch.empty((0,), dtype=torch.from_numpy(np.array([], dtype=tensor_dtype)).dtype)
            else:
                return torch.from_numpy(np.stack(data).astype(tensor_dtype))
        else:
            return torch.from_numpy(np.asarray(data, dtype=tensor_dtype))

    def _validate_shape(self, data, expected_shape_template, shape_context):
        """Validate data shapes against template using known dimensions."""
        # Parse expected shape template like "[S, H, W]" or "[N, 3]"
        expected_dims = expected_shape_template.strip("[]").split(", ")

        if isinstance(data, list):
            # For lists, validate that we have S elements and each has correct shape
            if len(expected_dims) < 2:
                raise ValueError(f"Invalid shape template for list data: {expected_shape_template}")

            expected_list_len = shape_context.get(expected_dims[0])
            if expected_list_len is not None and len(data) != expected_list_len:
                raise ValueError(f"Expected {expected_list_len} elements in list, got {len(data)}")

            # Validate shape of each array in the list (align by position)
            for i, item in enumerate(data):
                if isinstance(item, np.ndarray):
                    # Compare known expected dims by their position within the item (after list dim)
                    for j, dim in enumerate(expected_dims[1:]):
                        expected_dim = None
                        if dim.isdigit():
                            expected_dim = int(dim)
                        elif dim in shape_context:
                            expected_dim = shape_context[dim]
                        # Skip unknown dims
                        if expected_dim is None:
                            continue
                        if j < len(item.shape) and item.shape[j] != expected_dim:
                            raise ValueError(
                                f"List element [{i}] dimension {j}: expected {expected_dim}, got {item.shape[j]}"
                            )

        elif isinstance(data, np.ndarray):
            # For single arrays, validate shape directly
            for i, dim in enumerate(expected_dims):
                expected_dim = None
                if dim.isdigit():
                    expected_dim = int(dim)
                elif dim in shape_context:
                    expected_dim = shape_context[dim]
                # Skip unknown dimensions
                if expected_dim is None:
                    continue
                if i < len(data.shape) and data.shape[i] != expected_dim:
                    raise ValueError(f"Dimension {i}: expected {expected_dim}, got {data.shape[i]}")

    def _build_sample(self, batch):
        # Raw batch has been retrieved from the base dataset
        seq_name = batch[DataField.SEQ_NAME]
        is_synthetic = batch.get(DataField.IS_SYNTHETIC, False)
        metric_scale = batch.get(DataField.METRIC_SCALE, False)

        # First extract shape context from images (always required)
        images_data = batch.get(DataField.IMAGES)
        if images_data is None or not isinstance(images_data, list) or len(images_data) == 0:
            raise ValueError("IMAGES field is required and must be a non-empty list")

        S = len(images_data)  # Number of frames
        H, W = images_data[0].shape[:2]  # Height, Width from first image
        shape_context = {"S": S, "H": H, "W": W}

        # Validate and convert images with range checking
        images = self._validate_and_convert_field(
            images_data,
            required=True,
            expected_dtype=np.uint8,
            tensor_dtype=np.float32,
            validate_range=(0, 255),
            expected_shape_template="[S, H, W, 3]",
            shape_context=shape_context,
        )
        # [S, H, W, C] uint8-as-float32 -> [S, C, H, W] in [0, 1]
        images = images.movedim(-1, 1).contiguous()
        images.mul_(1.0 / 255.0)

        # Validate and convert other fields in one step
        depths = self._validate_and_convert_field(
            batch.get(DataField.DEPTHS),
            expected_dtype=np.float32,
            expected_shape_template="[S, H, W]",
            shape_context=shape_context,
        )
        extrinsics_c2w = self._validate_and_convert_field(
            batch.get(DataField.EXTRINSICS_C2W),
            expected_dtype=np.float32,
            expected_shape_template="[S, 4, 4]",
            shape_context=shape_context,
        )
        intrinsics = self._validate_and_convert_field(
            batch.get(DataField.INTRINSICS),
            expected_dtype=np.float32,
            expected_shape_template="[S, 3, 3]",
            shape_context=shape_context,
        )
        world_points = self._validate_and_convert_field(
            batch.get(DataField.WORLD_POINTS),
            expected_dtype=np.float32,
            expected_shape_template="[S, H, W, 3]",
            shape_context=shape_context,
        )
        point_masks = self._validate_and_convert_field(
            batch.get(DataField.POINT_MASKS),
            expected_dtype=bool,
            expected_shape_template="[S, H, W]",
            shape_context=shape_context,
            required_when=batch.get(DataField.EXTRINSICS_C2W) is not None
            and batch.get(DataField.WORLD_POINTS) is not None,
        )
        ids = self._validate_and_convert_field(
            batch.get(DataField.IDS),
            expected_dtype=np.int64,
            expected_shape_template="[S]",
            shape_context=shape_context,
            stack=False,
        )

        # Appearance augmentation (training only). The underlying parser
        # signals whether one transform should be drawn and applied to every
        # frame (CONSISTENT_AUG=True) or whether each frame should draw its
        # own transform (default).
        if self.training and self.image_aug is not None:
            images = self._apply_appearance_aug(
                images,
                consistent=bool(batch.get(DataField.CONSISTENT_AUG, False)),
            )

        # Invalidate all points if first frame has no valid points
        if self.training and point_masks is not None and point_masks.numel() > 0 and point_masks[0].sum() == 0:
            point_masks[:] = False

        # Scene normalization
        if self.normalize_scene and extrinsics_c2w is not None and world_points is not None:
            transform_matrix, scale_factor = compute_normalization_transform(extrinsics_c2w, world_points, point_masks)
            extrinsics_c2w = transform_matrix @ extrinsics_c2w
            extrinsics_c2w[:, :3, 3] = extrinsics_c2w[:, :3, 3] / scale_factor

            R = transform_matrix[:3, :3]
            t = transform_matrix[:3, 3]
            world_points = torch.einsum("shwi,ij->shwj", world_points, R.T) + t.view(1, 1, 1, 3)
            world_points = world_points / scale_factor
            depths = depths / scale_factor
        else:
            scale_factor = 1.0
            transform_matrix = torch.eye(4)

        candidate_fields: list[Tuple[Any, Any]] = [
            (DataField.SEQ_NAME, seq_name),
            (DataField.IMAGES, images),
            (DataField.IS_SYNTHETIC, torch.tensor([is_synthetic], dtype=torch.bool)),
            (DataField.METRIC_SCALE, torch.tensor([metric_scale], dtype=torch.bool)),
            (DataField.IDS, ids),
            (DataField.DEPTHS, depths),
            (DataField.EXTRINSICS_C2W, extrinsics_c2w),
            (DataField.INTRINSICS, intrinsics),
            (DataField.WORLD_POINTS, world_points),
            (DataField.POINT_MASKS, point_masks),
            (DataField.SCALE_FACTOR, torch.tensor(scale_factor)),
        ]
        sample = {key: value for key, value in candidate_fields if value is not None}

        missing = [field for field in self.load_data_fields if field not in sample]
        if missing:
            raise ValueError(f"Fields {missing} not found in sample after default fill")

        return sample

    def _apply_appearance_aug(self, images: torch.Tensor, *, consistent: bool) -> torch.Tensor:
        """Run :attr:`image_aug` over a ``(S, C, H, W)`` stack of frames.

        When ``consistent`` is True a single transform is drawn and applied to
        the whole stack at once; otherwise each frame draws its own transform.
        Both branches return a ``(S, C, H, W)`` tensor. The non-consistent
        branch unbinds the frames, applies the transform to each in turn, and
        re-stacks them; this is intentionally not the in-place
        ``images[i] = aug(images[i])`` mutation pattern used by other
        multi-view training stacks.
        """
        if consistent:
            return self.image_aug(images)
        per_frame = [self.image_aug(frame) for frame in images.unbind(dim=0)]
        return torch.stack(per_frame, dim=0)
