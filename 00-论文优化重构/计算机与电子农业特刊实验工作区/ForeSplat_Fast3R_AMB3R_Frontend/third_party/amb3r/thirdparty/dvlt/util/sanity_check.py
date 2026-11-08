# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock components for sanity check mode."""

import math
from numbers import Number
from typing import Any

import numpy as np
import torch
from accelerate.logging import get_logger


logger = get_logger(__name__)


class MockWandbTracker:
    """Mock wandb tracker for sanity check mode."""

    def __init__(self, *args, **kwargs):
        self.name = "wandb"
        self.run = self._MockRun()

    class _MockRun:
        def __init__(self):
            self.id = "mock_run_id"

    @staticmethod
    def _is_float_like(value: Any) -> bool:
        # Torch tensor: only float dtypes
        if isinstance(value, torch.Tensor):
            return torch.is_floating_point(value)
        # Numpy array or scalar: float dtypes
        if isinstance(value, (np.ndarray, np.generic)):
            try:
                dtype = value.dtype
                return np.issubdtype(dtype, np.floating)
            except Exception:
                return False
        # Python float
        return isinstance(value, float)

    @staticmethod
    def _is_finite(value: Any) -> bool:
        # Torch tensor
        if isinstance(value, torch.Tensor):
            return torch.isfinite(value).all().item()
        # Numpy array or scalar
        if isinstance(value, np.ndarray):
            return np.isfinite(value).all()
        if isinstance(value, np.generic):
            return np.isfinite(value).item()
        # Python numbers
        if isinstance(value, Number):
            return math.isfinite(float(value))
        # Non-numeric types are considered OK (we don't check them)
        return True

    def log(self, data, step=None):
        """Mock log method that prints key and type of value, and asserts finite losses."""
        if isinstance(data, dict):
            # Assert: any float-like metric value must be finite
            bad_keys = []
            for k, v in data.items():
                if self._is_float_like(v) and not self._is_finite(v):
                    bad_keys.append(k)
            if bad_keys:
                raise ValueError(
                    f"[SANITY CHECK] Non-finite values logged for keys: {', '.join(map(str, bad_keys))} at step {step}"
                )

            pairs = [f"{k} -> {type(v).__name__}" for k, v in data.items()]
            logger.info(f"[MOCK WANDB log] Step {step}: {', '.join(pairs)}")

    def log_images(self, data, step=None):
        """Mock log_images method that prints key and type of value."""
        if isinstance(data, dict):
            pairs = [f"{k} -> {type(v).__name__}" for k, v in data.items()]
            logger.info(f"[MOCK WANDB log_images] Step {step}: {', '.join(pairs)}")

    def finish(self):
        """Mock finish method called by accelerator during end_training."""
        logger.info("[MOCK WANDB finish] Mock tracker finished")


class MockAcceleratorTrackers:
    """Mock accelerator trackers list for sanity check mode."""

    def __init__(self):
        self._trackers = [MockWandbTracker()]

    def __iter__(self):
        return iter(self._trackers)

    def __getitem__(self, index):
        return self._trackers[index]

    def __len__(self):
        return len(self._trackers)
