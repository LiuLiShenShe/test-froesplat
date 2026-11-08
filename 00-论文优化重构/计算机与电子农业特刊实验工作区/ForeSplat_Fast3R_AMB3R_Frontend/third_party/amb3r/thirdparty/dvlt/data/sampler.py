# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in
# LICENSES/VGGT-LICENSE.txt in the root of this source tree.
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from collections import deque
from typing import Iterator, List, Optional, Tuple

import torch
from accelerate.logging import get_logger
from rich.table import Table
from torch.utils.data import Dataset, DistributedSampler, Sampler

from dvlt.util.console import render_table_to_text


logger = get_logger(__name__)


class AdaptiveBatchSampler(Sampler):
    """
    Dynamic batch sampler that adjusts batch size, aspect ratio, and image number
    for each batch based on randomly generated parameters.

    This sampler generates batches where each batch has a consistent aspect ratio
    and image number, but these parameters vary between batches. The batch size
    is calculated as max_frames_per_gpu // image_num to respect GPU memory limits.

    Additionally, if max_tokens_per_gpu is set, the sampler will optimize the batch
    size to maximize token utilization.

    Args:
        dataset: The dataset to sample from
        aspect_ratio_bounds: [min, max] range for aspect ratio generation
        frames_per_sample_range: [min, max] range for number of images per sample
        max_frames_per_gpu: Maximum number of images that can fit in GPU memory
        max_tokens_per_gpu: Maximum number of tokens that can fit in GPU memory.
                           If None, uses simpler mode without token optimization.
        image_size: Base image size (default: 512)
        patch_size: Vision transformer patch size (default: 16)
        shuffle: Whether to shuffle the dataset indices
        seed: Random seed for reproducible sampling
        epoch: Initial epoch number for seeding
        drop_last: Whether to drop the last incomplete batch (default: False)
        infinite: Whether to repeat the dataset indices indefinitely (default: True)
            This setting still uses epochs but avoids the dataloader iterator from
            being exhausted, so that the prefetch queue is never empty.
    """

    def __init__(
        self,
        dataset: Dataset,
        aspect_ratio_bounds: List[float],
        frames_per_sample_range: List[int],
        max_frames_per_gpu: int = 48,
        max_tokens_per_gpu: Optional[int] = None,
        image_size: int = 512,
        patch_size: int = 16,
        shuffle: bool = True,
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = False,
        infinite: bool = True,
    ) -> None:
        self.dataset = dataset
        self.aspect_ratio_bounds = aspect_ratio_bounds
        self.frames_per_sample_range = frames_per_sample_range
        self.max_frames_per_gpu = max_frames_per_gpu
        self.image_size = image_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.seed = seed
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.drop_last = drop_last
        self.infinite = infinite

        # RNG for parameter generation and per-sample seeds. Each dataset sample
        # receives a deterministic seed derived from this RNG, making in-worker
        # randomness (view selection, frame selection, augmentation) reproducible
        # across checkpoint/resume cycles.
        self.param_rng = random.Random()

        # Prefetch-aware state tracking: the DataLoader prefetches batches, advancing
        # the sampler state ahead of what the training loop has actually consumed.
        # We snapshot state after each yielded batch and "commit" when the training
        # loop consumes it, so state_dict() always reflects training loop position.
        self._state_snapshots: deque = deque()
        self._committed_state: Optional[dict] = None

        self.set_epoch(epoch)

        # Print out dataset distribution
        if len(self.cumulative_sizes) > 1:
            # Create a table showing dataset distribution
            table = Table(title="Dataset Distribution", show_header=True, header_style="bold magenta")
            table.add_column("Dataset", style="cyan", justify="center")
            table.add_column("Size", style="green", justify="right")
            table.add_column("Repeat", style="blue", justify="right")
            table.add_column("Samples/Epoch", style="purple", justify="right")
            table.add_column("Epoch %", style="yellow", justify="right")

            dataset_sizes = [
                size - self.cumulative_sizes[i - 1] if i > 0 else size for i, size in enumerate(self.cumulative_sizes)
            ]

            # Calculate samples per epoch based on repeat factors
            if self.repeat_factors is not None:
                samples_per_epoch = [
                    int(size * rf) for size, rf in zip(dataset_sizes, self.repeat_factors)
                ]
            else:
                samples_per_epoch = dataset_sizes
            total_samples = sum(samples_per_epoch)

            for i, size in enumerate(dataset_sizes):
                repeat_factor = self.repeat_factors[i] if self.repeat_factors is not None else 1.0
                samples = samples_per_epoch[i]
                proportion = (samples / total_samples) * 100 if total_samples > 0 else 0
                table.add_row(
                    f"{self.dataset.names[i] if hasattr(self.dataset, 'names') else i}",
                    f"{size:,}",
                    f"×{repeat_factor:.2f}",
                    f"{samples:,}",
                    f"{proportion:.2f}%",
                )

            # Add total row
            table.add_section()
            table.add_row(
                "TOTAL",
                f"{sum(dataset_sizes):,}",
                "-",
                f"{total_samples:,}",
                "100.00%",
                style="bold",
            )

            logger.info(render_table_to_text(table))

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for reproducible sampling.

        This method should be called at the beginning of each epoch to ensure
        proper randomization while maintaining reproducibility across runs.

        Always clears indices to start the epoch fresh. For mid-epoch resume,
        use load_state_dict() instead, which restores saved indices.

        Args:
            epoch: The current epoch number
        """
        self.epoch = epoch
        self.indices = []
        self.param_rng.seed(self.seed + epoch)
        self._state_snapshots.clear()

    def commit_batch(self):
        """Mark the oldest prefetched batch as consumed by the training loop.

        Called by the dataloader after each batch is yielded to the user.
        This keeps the committed state in sync with actual training progress,
        regardless of how many batches the DataLoader has prefetched ahead.
        """
        if self._state_snapshots:
            self._committed_state = self._state_snapshots.popleft()

    def state_dict(self):
        """
        Return state for checkpointing.

        Returns the last committed state (matching training loop position) rather
        than the sampler's current internal state, which may be ahead due to
        DataLoader prefetching. This prevents skipping batches on resume.

        Stores only the count of remaining indices rather than the full list,
        significantly reducing checkpoint size. Indices are deterministically
        regenerated on restore based on epoch and seed.
        """
        if self._committed_state is not None:
            return dict(self._committed_state)
        return {
            "num_remaining": len(self.indices),
            "epoch": self.epoch,
            "param_rng_state": self.param_rng.getstate(),
        }

    def load_state_dict(self, state_dict):
        """
        Load state from checkpoint.

        Regenerates the full indices list deterministically and truncates to
        the saved position.
        """
        self.epoch = state_dict.get("epoch", 0)

        # Restore param_rng state (critical for batch parameter and sample seed generation)
        self.param_rng.setstate(state_dict["param_rng_state"])

        num_remaining = state_dict.get("num_remaining", 0)
        if num_remaining > 0:
            # Generate full indices list deterministically
            # (uses epoch and seed internally)
            self._generate_indices()

            # Truncate to keep only remaining indices
            # Since we pop from the end, remaining indices are at the front
            self.indices = self.indices[:num_remaining]
        else:
            self.indices = []

    @property
    def repeat_factors(self) -> Optional[List[float]]:
        if hasattr(self.dataset, "repeat_factors"):
            return self.dataset.repeat_factors
        else:
            return None

    @property
    def cumulative_sizes(self) -> List[int]:
        if hasattr(self.dataset, "cumulative_sizes"):
            return self.dataset.cumulative_sizes
        else:
            return [len(self.dataset)]

    def _generate_parameters(self) -> Tuple[int, float, int]:
        """
        Generate dynamic parameters for a batch.

        Returns:
            Tuple of (image_num, aspect_ratio, batch_size) for the current batch
        """
        image_num = self.param_rng.randint(self.frames_per_sample_range[0], self.frames_per_sample_range[1])
        aspect_ratio = round(self.param_rng.uniform(self.aspect_ratio_bounds[0], self.aspect_ratio_bounds[1]), 2)

        # Calculate initial batch_size from max_frames_per_gpu
        batch_size = max(1, self.max_frames_per_gpu // image_num)

        # If max_tokens_per_gpu is not set, use old behavior (no token optimization)
        if self.max_tokens_per_gpu is None:
            return image_num, aspect_ratio, batch_size

        # Token-based optimization
        # Calculate tokens per image for this aspect ratio
        short_size = int(self.image_size * aspect_ratio)
        if short_size % self.patch_size != 0:
            short_size = (short_size // self.patch_size) * self.patch_size
        tokens_per_image = (self.image_size // self.patch_size) * (short_size // self.patch_size)

        # Optimize batch_size and image_num to maximize token utilization
        current_tokens = batch_size * image_num * tokens_per_image

        # If we exceed max tokens, reduce batch_size (prefer reducing batch_size over image_num)
        if current_tokens > self.max_tokens_per_gpu:
            batch_size = max(1, self.max_tokens_per_gpu // (image_num * tokens_per_image))
            current_tokens = batch_size * image_num * tokens_per_image

        # Try to increase batch_size to better utilize available tokens
        # Only increase if we can fit at least one more full batch item
        elif current_tokens < self.max_tokens_per_gpu:
            optimal_batch_size = self.max_tokens_per_gpu // (image_num * tokens_per_image)
            # Only increase batch_size if it improves utilization by at least 10%
            if optimal_batch_size > batch_size:
                new_tokens = optimal_batch_size * image_num * tokens_per_image
                if new_tokens <= self.max_tokens_per_gpu:
                    batch_size = optimal_batch_size

        return image_num, aspect_ratio, batch_size

    def _generate_indices(self) -> None:
        """
        Generate and store dataset indices for the current epoch.

        When repeat_factors are provided, each value determines
        how many times samples from that dataset are seen per epoch:
        - repeat_factor = 1.0: each sample seen once (natural distribution)
        - repeat_factor = 2.0: each sample seen twice (oversampling)
        - repeat_factor = 0.5: half the samples seen once (undersampling)
        - repeat_factor = 0.0: skip this dataset entirely

        Samples are drawn using repeated permutations for oversampling and
        random subsampling for undersampling.

        Stores indices in self.indices for consumption via popping.

        Uses a local generator seeded by epoch and seed for deterministic
        index generation, enabling efficient checkpoint/resume.
        """
        # Create local generator for deterministic index generation
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        total_size = len(self.dataset)

        if self.repeat_factors is not None and len(self.cumulative_sizes) > 1:
            cumulative_sizes = torch.LongTensor(self.cumulative_sizes)
            dataset_low = torch.cat([torch.zeros(1, dtype=torch.long), cumulative_sizes[:-1]])

            all_indices = []
            for i, repeat_factor in enumerate(self.repeat_factors):
                if repeat_factor == 0:
                    continue

                low = dataset_low[i].item()
                high = cumulative_sizes[i].item()
                base_indices = torch.arange(low, high)
                dataset_size = len(base_indices)

                # repeat_factor directly controls repetition
                full_repeats = int(repeat_factor)
                fractional_part = repeat_factor - full_repeats

                dataset_indices = []

                # Add full repetitions, each with different random permutation
                for _ in range(full_repeats):
                    perm = torch.randperm(dataset_size, generator=generator)
                    dataset_indices.append(base_indices[perm])

                # Add fractional part as random subsample
                if fractional_part > 0:
                    subsample_size = int(dataset_size * fractional_part)
                    perm = torch.randperm(dataset_size, generator=generator)
                    dataset_indices.append(base_indices[perm[:subsample_size]])

                if dataset_indices:
                    all_indices.append(torch.cat(dataset_indices))

            if not all_indices:
                raise ValueError("No indices to sample from")
            indices = torch.cat(all_indices)
        else:
            indices = torch.arange(total_size, dtype=torch.long)

        if self.shuffle:
            # Convert to torch tensor for randperm
            indices = indices[torch.randperm(len(indices), generator=generator)]

        # Reverse so that pop() yields indices in correct order (0, 1, 2, ... for shuffle=False)
        self.indices = indices.tolist()[::-1]

    def _create_batches(self) -> Iterator[List[Tuple[int, int, float, int]]]:
        """
        Create batches by popping indices from self.indices.

        Each batch contains tuples of (index, image_num, aspect_ratio, sample_seed)
        where all tuples in a batch share the same image_num and aspect_ratio.
        The sample_seed makes in-worker randomness deterministic across resume.

        After each yielded batch, a state snapshot is pushed so that
        commit_batch() can track which batches the training loop has consumed.

        Yields:
            Batches as lists of (index, image_num, aspect_ratio, sample_seed) tuples
        """
        while self.indices:
            image_num, aspect_ratio, batch_size = self._generate_parameters()

            batch = []
            for _ in range(batch_size):
                if not self.indices:
                    break
                idx = self.indices.pop()
                sample_seed = self.param_rng.randint(0, 2**31 - 1)
                batch.append((idx, image_num, aspect_ratio, sample_seed))

            # Yield batch if it's complete, or if drop_last=False and it's non-empty
            if len(batch) == batch_size or (not self.drop_last and batch):
                self._state_snapshots.append(
                    {
                        "num_remaining": len(self.indices),
                        "epoch": self.epoch,
                        "param_rng_state": self.param_rng.getstate(),
                    }
                )
                yield batch

    def __iter__(self) -> Iterator[List[Tuple[int, int, float]]]:
        """
        Iterate over batches for one epoch.

        If indices were restored via load_state_dict() (mid-epoch resume),
        continues from where it left off. Otherwise generates fresh indices.

        Auto-increments epoch after finishing all batches to ensure proper
        RNG seeding for the next epoch.

        If infinite is True, the sampler will continue to yield batches indefinitely.

        Yields:
            Batches as lists of (index, image_num, aspect_ratio, sample_seed) tuples
        """
        while True:
            # Generate indices if not already present (e.g., from load_state_dict)
            if not self.indices:
                self._generate_indices()

            # Yield all batches for this epoch
            yield from self._create_batches()

            # Auto-increment epoch after completing all batches
            # This ensures the next iteration uses a different RNG seed
            self.set_epoch(self.epoch + 1)

            if not self.infinite:
                break

    def __len__(self) -> int:
        """Approximate number of batches (batch sizes vary dynamically)."""
        avg_image_num = (self.frames_per_sample_range[0] + self.frames_per_sample_range[1]) / 2
        avg_batch_size = max(1, self.max_frames_per_gpu // avg_image_num)
        dataset_size = len(self.dataset)
        batch_size = int(avg_batch_size)
        if self.drop_last:
            return max(1, dataset_size // batch_size)
        else:
            return max(1, (dataset_size + batch_size - 1) // batch_size)


class DistributedAdaptiveBatchSampler(AdaptiveBatchSampler, DistributedSampler):
    """
    Dynamic batch sampler for distributed training.

    Combines the functionality of AdaptiveBatchSampler with DistributedSampler
    to enable dynamic batching in multi-GPU distributed training scenarios.

    This sampler ensures that:
    - Each GPU gets a disjoint subset of the dataset (via DistributedSampler)
    - Dynamic parameters are synchronized across all GPUs
    - When max_tokens_per_gpu is set: batch sizes and image_num are optimized
      to maximize GPU token utilization
    - When max_tokens_per_gpu is None: uses simpler batch_size = max_frames_per_gpu // image_num

    Args:
        dataset: The dataset to sample from
        aspect_ratio_bounds: [min, max] range for aspect ratio generation
        frames_per_sample_range: [min, max] range for number of images per sample
        max_frames_per_gpu: Maximum number of images that can fit in GPU memory
        max_tokens_per_gpu: Maximum number of tokens that can fit in GPU memory.
                           If None, uses simpler mode without token optimization.
        image_size: Base image size (default: 512)
        patch_size: Vision transformer patch size (default: 16)
        num_replicas: Number of processes participating in distributed training.
                     If None, uses torch.distributed.get_world_size()
        rank: Rank of the current process. If None, uses torch.distributed.get_rank()
        shuffle: Whether to shuffle the dataset indices
        seed: Random seed for reproducible sampling
        drop_last: Whether to drop the last incomplete batch
        epoch: Initial epoch number for seeding
        infinite: Whether to repeat the dataset indices indefinitely (default: True)
            This setting still uses epochs but avoids the dataloader iterator from
            being exhausted, so that the prefetch queue is never empty.
    """

    def __init__(
        self,
        dataset: Dataset,
        aspect_ratio_bounds: List[float],
        frames_per_sample_range: List[int],
        max_frames_per_gpu: int = 48,
        max_tokens_per_gpu: Optional[int] = None,
        image_size: int = 512,
        patch_size: int = 16,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        epoch: int = 0,
        infinite: bool = True,
    ) -> None:
        # Initialize AdaptiveBatchSampler
        AdaptiveBatchSampler.__init__(
            self,
            dataset,
            aspect_ratio_bounds,
            frames_per_sample_range,
            max_frames_per_gpu,
            max_tokens_per_gpu,
            image_size,
            patch_size,
            shuffle,
            seed,
            epoch,
            drop_last,
            infinite,
        )

        # Initialize DistributedSampler
        DistributedSampler.__init__(self, dataset, num_replicas, rank, shuffle, seed, drop_last)

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for both dynamic and distributed functionality.

        This ensures that both the dynamic parameter generation and the
        distributed sampling are properly seeded for the given epoch.

        Args:
            epoch: The current epoch number
        """
        AdaptiveBatchSampler.set_epoch(self, epoch)
        DistributedSampler.set_epoch(self, epoch)

    def state_dict(self):
        """Return state for checkpointing (delegates to parent class)."""
        return AdaptiveBatchSampler.state_dict(self)

    def load_state_dict(self, state_dict):
        """Load state from checkpoint (delegates to parent class)."""
        AdaptiveBatchSampler.load_state_dict(self, state_dict)

    def _generate_indices(self) -> None:
        """
        Generate distributed indices for this rank.

        For non-weighted sampling, uses DistributedSampler's built-in logic.
        For weighted sampling, generates weighted indices then distributes.
        """
        if self.repeat_factors is None or len(self.cumulative_sizes) <= 1:
            # Use DistributedSampler's built-in logic (handles padding and distribution)
            self.indices = list(DistributedSampler.__iter__(self))
            # Reverse so that pop() yields indices in correct order (consistent with parent class)
            self.indices = self.indices[::-1]
        else:
            # Weighted sampling: generate indices then distribute
            AdaptiveBatchSampler._generate_indices(self)

            # Pad to make evenly divisible by num_replicas
            total_size = len(self.indices)
            num_samples_per_rank = (total_size + self.num_replicas - 1) // self.num_replicas
            padded_size = num_samples_per_rank * self.num_replicas

            if padded_size > total_size:
                self.indices = self.indices + self.indices[: padded_size - total_size]

            # Distribute across ranks
            self.indices = self.indices[self.rank :: self.num_replicas]
