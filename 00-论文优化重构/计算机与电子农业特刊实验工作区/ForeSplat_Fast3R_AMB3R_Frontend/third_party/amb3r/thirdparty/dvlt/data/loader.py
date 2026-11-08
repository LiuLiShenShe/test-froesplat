# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from accelerate.data_loader import DataLoaderAdapter, DataLoaderStateMixin
from accelerate.state import GradientState
from accelerate.utils import send_to_device


class AccelerateDataLoader(DataLoaderAdapter, DataLoaderStateMixin):
    """
    A DataLoader that provides automatic device placement like Accelerate's DataLoaderShard
    but without any sharding logic to handle custom distributed samplers.

    This class provides:
    - Automatic device placement using Accelerate's send_to_device()
    - Compatible with mixed precision, FSDP, and other Accelerate features

    Unlike Accelerate's prepare() method, this loader doesn't add sharding logic,
    allowing custom distributed samplers to handle distribution themselves.
    """

    def __init__(
        self,
        dataset,
        device: Optional[torch.device] = None,
        non_blocking: bool = False,
        use_stateful_dataloader: bool = False,
        **kwargs,
    ):
        """Initialize the DataLoader.

        Args:
            dataset: The dataset to load from.
            device: The device to move the data to.
            non_blocking: If set to `True`, dataloader will utilize non-blocking host-to-device transfers.
            use_stateful_dataloader: Whether to use StatefulDataLoader for better state management.
            **kwargs: Additional arguments to pass to the DataLoader.
        """
        # Initialize DataLoaderAdapter (handles stateful dataloader and state dict)
        DataLoaderAdapter.__init__(self, dataset, use_stateful_dataloader=use_stateful_dataloader, **kwargs)

        # Initialize DataLoaderStateMixin (handles gradient state and end-of-dataloader tracking)
        self.gradient_state = GradientState()

        # Store device placement settings
        self.device = device
        self.non_blocking = non_blocking

        # Store references for easy access
        self.dataset = dataset
        self.batch_sampler = kwargs.get("batch_sampler", None)

    def __iter__(self):
        """Iterate with automatic device placement and proper state management."""
        # Begin gradient state tracking like DataLoaderShard
        self.begin()

        for batch in self.base_dataloader:
            if self.device is not None:
                # Use Accelerate's send_to_device for proper device placement
                batch = send_to_device(batch, self.device, non_blocking=self.non_blocking)

            # Update state dict for proper resume functionality
            self._update_state_dict()

            # Mark this batch as consumed by the training loop so the sampler's
            # committed state stays in sync with actual progress (not prefetch position)
            if self.batch_sampler is not None and hasattr(self.batch_sampler, "commit_batch"):
                self.batch_sampler.commit_batch()

            yield batch

        self.end()

    def __len__(self):
        return len(self.base_dataloader)

    def __reduce__(self):
        """Define __reduce__ for proper pickling support."""
        args = super().__reduce__()
        return (AccelerateDataLoader, *args[1:])
