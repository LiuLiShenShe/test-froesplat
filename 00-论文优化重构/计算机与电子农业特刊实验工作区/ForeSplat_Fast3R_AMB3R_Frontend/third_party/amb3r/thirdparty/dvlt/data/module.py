# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import random
from functools import partial
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.eval import EvalDataset
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.train import TrainDataset
from dvlt.data.loader import AccelerateDataLoader
from dvlt.data.sampler import AdaptiveBatchSampler, DistributedAdaptiveBatchSampler


def _worker_init_fn(worker_id: int, rank: int = 0) -> None:
    """Seed all RNGs per worker for reproducibility in distributed training.

    PyTorch automatically seeds torch per worker (base_seed + worker_id), but
    this doesn't account for distributed rank. We add rank offset to make seeds
    unique across GPUs.
    cf. https://docs.pytorch.org/docs/stable/data.html#data-loading-randomness

    Args:
        worker_id: The worker ID (0 to num_workers-1), provided by DataLoader.
        rank: The global rank in distributed training (default 0 for single-GPU).
    """
    # Multiply rank by large offset to avoid collision with worker_id increments
    worker_seed = (torch.initial_seed() + rank * 10000) % (2**32)
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DataModule:
    """Data module for managing dataset loading and processing."""

    def __init__(
        self,
        train_datasets: Dict[str, TrainDataset] | None = None,
        train_config: Dict[str, Any] | None = None,
        test_datasets: Dict[str, EvalDataset] | None = None,
        test_config: Dict[str, Any] | None = None,
        image_size: int = 518,
        patch_size: int = 14,
        tokens_per_batch: int | None = None,
        images_per_batch: int = 48,
        images_per_element: Tuple[int, int] | int = (2, 24),
        aspect_ratios: Tuple[float, float] | float = (0.33, 1.0),
        train_num_workers: int = 2,
        train_prefetch_factor: int = 2,
        test_num_workers: int = 2,
        collate_fn: Callable | str = default_collate_fn,
        infinite_sampling: bool = True,
        distributed_eval: bool = True,
        pin_memory: bool = False,
    ):
        """Initialize the data module with configuration options.

        Args:
            train_datasets: Dict of training datasets with dataset names as keys.
            train_config: MultiSourceDataset configuration for training.
            test_datasets: Dict of testing datasets with dataset names as keys.
            test_config: MultiSourceDataset configuration for testing.
            image_size: Image size (longest side). Default is 518.
            patch_size: Patch size (i.e. factor by which image dimensions must be divisible). Default is 14.
            tokens_per_batch: Maximum number of tokens that can fit in GPU memory. If None, no token optimization is performed.
            images_per_batch: Number of images per batch.
            images_per_element: Number of images per batch element (min, max) or a single value.
            aspect_ratios: Aspect ratios for the training data (min, max) or a single value.
            train_num_workers: Number of subprocesses to use for data loading during training.
            train_prefetch_factor: Number of samples to prefetch for data loading during training.
            test_num_workers: Number of subprocesses to use for data loading during testing.
            collate_fn: Function to collate samples into a batch. Can be a callable or a string reference.
            infinite_sampling: Whether to sample infinitely or stop after one epoch. This is to keep
            the prefetch queue from being empty. Default is True.
            distributed_eval: Whether to distribute evaluation across GPUs. When False, the GPUs
                process all samples independently and only rank 0 results are logged. This avoids
                NCCL deadlocks when validation datasets are smaller than the number of GPUs.
                Default is True.
            pin_memory: Whether to pin memory for faster host-to-device transfers during training. Default is False.
        """
        # if not single value, it should be a tuple
        if not isinstance(images_per_element, int):
            images_per_element = tuple(images_per_element)
        else:
            images_per_element = (images_per_element, images_per_element)
        if not isinstance(aspect_ratios, float):
            aspect_ratios = tuple(aspect_ratios)
        else:
            aspect_ratios = (aspect_ratios, aspect_ratios)

        # Validate the aspect ratio and image number ranges
        if len(aspect_ratios) != 2 or aspect_ratios[0] > aspect_ratios[1]:
            raise ValueError(f"aspect_ratios must be [min, max] with min <= max, got {aspect_ratios}")
        if len(images_per_element) != 2 or images_per_element[0] < 1 or images_per_element[0] > images_per_element[1]:
            raise ValueError(f"images_per_element must be [min, max] with 1 <= min <= max, got {images_per_element}")
        if max(images_per_element) > images_per_batch:
            raise ValueError(
                f"images_per_element must be <= images_per_batch, got {max(images_per_element)} > {images_per_batch}"
            )

        self.train_datasets = train_datasets
        self.train_config = train_config if train_config is not None else {}
        self.test_datasets = test_datasets
        self.test_config = test_config if test_config is not None else {}
        self.tokens_per_batch = tokens_per_batch
        self.images_per_batch = images_per_batch
        self.images_per_element = images_per_element
        self.aspect_ratios = aspect_ratios
        self.image_size = image_size
        self.patch_size = patch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor
        self.test_num_workers = test_num_workers
        self.infinite_sampling = infinite_sampling
        self.distributed_eval = distributed_eval
        self.pin_memory = pin_memory

        # Set image size, patch size, and name
        for name, train_dataset in self.train_datasets.items():
            train_dataset.set_image_params(image_size, patch_size)
            train_dataset.set_name(name)
        for name, test_dataset in self.test_datasets.items():
            test_dataset.set_image_params(image_size, patch_size)
            test_dataset.set_name(name)

        # Handle string reference to collate_fn
        if isinstance(collate_fn, str):
            module_path, function_name = collate_fn.rsplit(".", 1)
            module = __import__(module_path, fromlist=[function_name])
            self.collate_fn = getattr(module, function_name)
        else:
            self.collate_fn = collate_fn

    def train_dataloader(self, accelerator: Accelerator | None = None) -> DataLoader:
        assert self.train_datasets is not None, "train_datasets is not set"
        train_dataset = MultiSourceDataset(
            self.train_datasets,
            training=True,
            **self.train_config,
        )

        # Choose the appropriate sampler based on distributed training
        if accelerator is not None and accelerator.num_processes > 1:
            batch_sampler = DistributedAdaptiveBatchSampler(
                dataset=train_dataset,
                aspect_ratio_bounds=self.aspect_ratios,
                frames_per_sample_range=self.images_per_element,
                max_tokens_per_gpu=self.tokens_per_batch,
                max_frames_per_gpu=self.images_per_batch,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                infinite=self.infinite_sampling,
            )
        else:
            batch_sampler = AdaptiveBatchSampler(
                dataset=train_dataset,
                aspect_ratio_bounds=self.aspect_ratios,
                frames_per_sample_range=self.images_per_element,
                max_tokens_per_gpu=self.tokens_per_batch,
                max_frames_per_gpu=self.images_per_batch,
                image_size=self.image_size,
                patch_size=self.patch_size,
                infinite=self.infinite_sampling,
            )

        # Use AccelerateDataLoader for automatic device placement without sharding
        # This handles device placement while allowing our custom distributed sampler to handle distribution
        dataloader_kwargs = {
            "batch_sampler": batch_sampler,
            "shuffle": False,
            "collate_fn": self.collate_fn,
            "num_workers": self.train_num_workers,
            "pin_memory": self.pin_memory,
            "device": accelerator.device if accelerator is not None else None,
            "non_blocking": True,  # Enable non-blocking transfers for better performance
        }
        if self.train_num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.train_prefetch_factor
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["timeout"] = 600  # 10 min timeout to detect stalled workers before NCCL timeout
            rank = accelerator.process_index if accelerator is not None else 0
            dataloader_kwargs["worker_init_fn"] = partial(_worker_init_fn, rank=rank)

        train_dataloader = AccelerateDataLoader(train_dataset, **dataloader_kwargs)

        # Register batch_sampler for checkpointing (saves epoch, RNG state, remaining indices)
        if accelerator is not None:
            accelerator.register_for_checkpointing(batch_sampler)

        return train_dataloader

    def test_dataloaders(self, accelerator: Accelerator | None = None) -> Dict[str, DataLoader]:
        assert self.test_datasets is not None, "test_datasets is not set"
        test_dataloaders = {}
        for name, test_dataset in self.test_datasets.items():
            test_dataloader = DataLoader(
                MultiSourceDataset({name: test_dataset}, training=False, **self.test_config),
                batch_size=1,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.test_num_workers,
            )
            test_dataloaders[name] = test_dataloader
        # Only distribute evaluation if distributed_eval is True
        # When False, all GPUs process all samples independently to avoid NCCL deadlocks
        if accelerator is not None and self.distributed_eval:
            test_dataloaders = {k: accelerator.prepare(v) for k, v in test_dataloaders.items()}
        return test_dataloaders
