# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Base class for evaluation datasets in dvlt.

Concrete eval-time parsers (ScanNet++, ETH3D, ...) inherit from
:class:`EvalDataset`. Compared to :class:`dvlt.data.datasets.train.TrainDataset`,
the eval pipeline is non-augmented: it resizes to a fixed short side,
optionally pads or center-crops to a square, and respects whichever view
sampling / ranking the user picks (``index``, ``pose``, ``middle_first``;
``first``, ``uniform``, ``every_X``).
"""

from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dvlt.data.datasets.util import rank_views_by_pose_similarity


class EvalDataset(Dataset):
    """
    Dataset class for testing.

    Defines preprocessing and sampling logic for test datasets.
    """

    def __init__(
        self,
        max_frames: Optional[int] = None,
        view_ranking: str = "index",
        view_sampling: str = "uniform",
        mode: str = "crop",
        max_depth_thresh: Optional[float] = None,
    ):
        """
        Initialize the test dataset.

        Args:
            max_frames: Optional maximum number of frames to load
            view_ranking: View ranking method (index: Use frame indices [default]; pose: Rank view by pose similarity to first frame).
            view_sampling: View sampling method (first: Use first max_frames frames, uniform: Use max_frames uniformly
                sampled frames, every_X: Use every X-th frame (e.g. every_5 takes frames 0, 5, 10, ...).
            mode: Image processing mode (crop, pad).
            max_depth_thresh: Maximum depth threshold. All depths greater than this value are set to invalid.
                 If None, no thresholding is applied.
        """
        self.max_frames = max_frames
        self.img_size = None
        self.patch_size = None
        self.mode = mode
        self.view_ranking = view_ranking
        self.view_sampling = view_sampling
        self.max_depth_thresh = max_depth_thresh
        self.name = "null"

    def set_image_params(self, img_size: int, patch_size: int):
        """Set the image size and patch size."""
        self.img_size = img_size
        self.patch_size = patch_size

    def set_name(self, name: str):
        """Set the name of the dataset."""
        self.name = name

    def _preprocess_images_depths_intrinsics(
        self,
        images: list[np.ndarray],
        depths: Optional[list[np.ndarray]] = None,
        intrinsics: Optional[list[np.ndarray]] = None,
    ) -> tuple[list[np.ndarray], Optional[list[np.ndarray]], Optional[list[np.ndarray]]]:
        """Preprocess images, depths, and intrinsics for testing.

        Args:
            images: List of S images as numpy arrays with shape [H, W, 3] (RGB, uint8).
            depths: Optional list of S depth maps as numpy arrays with shape [H, W] (float32).
                   Must have the same length as images if provided.
            intrinsics: Optional list of S intrinsic camera matrices as numpy arrays with shape [3, 3] (float32).
                       Must have the same length as images if provided.

        Returns:
            Tuple of (processed_images, processed_depths, processed_intrinsics):
            - processed_images: List of numpy arrays with shape [H', W', 3] (float32, normalized to [0,1])
            - processed_depths: List of numpy arrays with shape [H', W'] (float32) or None
            - processed_intrinsics: List of numpy arrays with shape [3, 3] (float32) or None

            Where H' and W' are the processed image dimensions after resize/crop/pad operations.
            All outputs are spatially aligned and have consistent dimensions.

        Note:
            All inputs are transformed with the same resize/crop/pad operations to maintain spatial alignment.
            If images have different shapes, they will be padded to the largest dimensions after processing.
        """
        if len(images) == 0:
            raise ValueError("At least 1 image is required")

        if self.mode not in ["crop", "pad"]:
            raise ValueError("Mode must be either 'crop' or 'pad'")

        images_resized = []
        depths_resized = []
        intrinsics_resized = []
        shapes = set()

        depths = [None] * len(images) if depths is None else depths
        intrinsics = [None] * len(images) if intrinsics is None else intrinsics

        for i, img in enumerate(images):
            img_h, img_w = img.shape[:2]
            if depths[i] is not None:
                depth_h, depth_w = depths[i].shape[:2]
                assert (depth_h, depth_w) == (img_h, img_w), (
                    f"Frame {i}: Depth map spatial dimensions {(depth_h, depth_w)} "
                    f"must match image dimensions {(img_h, img_w)}"
                )

        for img, depth, intr in zip(images, depths, intrinsics):
            img = Image.fromarray(img)
            if img.mode == "RGBA":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            img = img.convert("RGB")

            width, height = img.size

            if self.mode == "pad":
                if width >= height:
                    new_width = self.img_size
                    new_height = round(height * (new_width / width) / self.patch_size) * self.patch_size
                else:
                    new_height = self.img_size
                    new_width = round(width * (new_height / height) / self.patch_size) * self.patch_size
            else:
                new_width = self.img_size
                new_height = round(height * (new_width / width) / self.patch_size) * self.patch_size

            scale_w = new_width / width
            scale_h = new_height / height

            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = np.array(img, dtype=np.uint8)

            if depth is not None:
                if self.max_depth_thresh is not None:
                    depth[depth > self.max_depth_thresh] = 0.0
                depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST).astype(np.float32)

            if intr is not None:
                intr = intr.astype(np.float32)
                intr[0, 0] = intr[0, 0] * scale_w
                intr[1, 1] = intr[1, 1] * scale_h
                intr[0, 2] = intr[0, 2] * scale_w
                intr[1, 2] = intr[1, 2] * scale_h

            if self.mode == "crop" and new_height > self.img_size:
                start_y = (new_height - self.img_size) // 2
                img = img[start_y : start_y + self.img_size, :, :]
                if depth is not None:
                    depth = depth[start_y : start_y + self.img_size, :]
                if intr is not None:
                    intr[1, 2] = intr[1, 2] - start_y

            if self.mode == "pad":
                h_padding = self.img_size - img.shape[0]
                w_padding = self.img_size - img.shape[1]
                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    img = np.pad(
                        img,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                    if depth is not None:
                        depth = np.pad(
                            depth, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0.0
                        )
                    if intr is not None:
                        intr[0, 2] = intr[0, 2] + pad_left
                        intr[1, 2] = intr[1, 2] + pad_top

            shapes.add((img.shape[0], img.shape[1]))
            images_resized.append(img)
            depths_resized.append(depth)
            intrinsics_resized.append(intr)

        if len(shapes) > 1:
            print(f"Warning: Found images with different shapes: {shapes}")
            max_height = max(shape[0] for shape in shapes)
            max_width = max(shape[1] for shape in shapes)

            padded_images = []
            padded_depths = []
            padded_intrinsics = []

            for img, depth, intr in zip(images_resized, depths_resized, intrinsics_resized):
                h_padding = max_height - img.shape[0]
                w_padding = max_width - img.shape[1]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    img = np.pad(
                        img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0
                    )
                    if depth is not None:
                        depth = np.pad(
                            depth, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0.0
                        )
                    if intr is not None:
                        intr[0, 2] = intr[0, 2] + pad_left
                        intr[1, 2] = intr[1, 2] + pad_top

                padded_images.append(img)
                padded_depths.append(depth)
                padded_intrinsics.append(intr)

            images_resized = padded_images
            depths_resized = padded_depths
            intrinsics_resized = padded_intrinsics

        depths_out = depths_resized if depths_resized and depths_resized[0] is not None else None
        intrinsics_out = intrinsics_resized if intrinsics_resized and intrinsics_resized[0] is not None else None

        return (images_resized, depths_out, intrinsics_out)

    def _get_indices(
        self, video_len: int, max_frames: Optional[int] = None, extrinsics_c2w: Optional[np.ndarray] = None
    ) -> list[int]:
        indices = np.arange(video_len, dtype=np.int32)
        if self.view_ranking == "pose":
            assert extrinsics_c2w is not None, "Extrinsics are required for pose ranking"
            indices = rank_views_by_pose_similarity(0, extrinsics_c2w)[indices]
        elif self.view_ranking == "middle_first":
            mid = video_len // 2
            indices = np.concatenate([[mid], indices[:mid], indices[mid + 1 :]])
        elif self.view_ranking != "index":
            raise ValueError(f"Invalid view ranking method: {self.view_ranking}")

        if self.view_sampling.startswith("every_"):
            assert self.max_frames is None, "Max frames must not be set for every_X sampling"
            step = int(self.view_sampling.split("_")[1])
            indices = indices[::step]
        elif max_frames is not None and max_frames < video_len:
            if self.view_sampling == "first":
                indices = indices[:max_frames]
            elif self.view_sampling == "uniform":
                indices = indices[np.linspace(0, video_len, max_frames, endpoint=False, dtype=np.int32)]
            else:
                raise ValueError(f"Invalid view sampling method: {self.view_sampling}")

        return indices.tolist()

    def __getitem__(self, idx_N: int | tuple[int, Optional[list[str]]]) -> dict[str, Any]:
        """
        Get an item from the dataset.

        Args:
            idx_N: Either an integer index or tuple containing (seq_index, data_fields)

        Returns:
            Dataset item as returned by get_data()
        """
        assert self.img_size is not None and self.patch_size is not None, "Image size and patch size must be set"
        assert self.img_size % self.patch_size == 0, "Image size must be divisible by patch size"
        if isinstance(idx_N, tuple):
            seq_index, data_fields = idx_N[0], idx_N[1]
        else:
            seq_index = idx_N
            data_fields = None
        return self.get_data(seq_index=seq_index, data_fields=data_fields)

    def get_data(self, seq_index: int, data_fields: Optional[list[str]] = None) -> dict[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def available_data_fields(self):
        raise NotImplementedError
