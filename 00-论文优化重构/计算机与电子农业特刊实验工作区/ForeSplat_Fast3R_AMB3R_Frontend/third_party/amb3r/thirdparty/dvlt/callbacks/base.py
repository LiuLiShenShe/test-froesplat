# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base callback class."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from dvlt.engine.trainer import Trainer


@dataclass
class CallbackConfig:
    """Base callback configuration."""

    _target_: str = "dvlt.callbacks.base.Callback"


class Callback:
    """Base class for callbacks.

    Callbacks provide hooks into the training and evaluation loops to perform
    custom actions at specific points.

    Training hooks:
      - on_train_batch: Called periodically during training for logging/visualization

    Evaluation hooks:
      - on_test_dataset_start: Called at the start of testing a dataset
      - on_test_batch: Called after processing a test batch
      - on_test_dataset_end: Called at the end of testing a dataset

    All methods receive trainer parameter for accessing trainer state and model.
    """

    def on_train_batch(
        self,
        batch: dict,
        predictions: dict,
        step: int,
        trainer: "Trainer",
    ) -> dict:
        """Called periodically during training for logging/visualization.

        Args:
            batch: Training batch (ground truth).
            predictions: Model predictions.
            step: Current training step.
            trainer: Trainer instance (access via trainer.model, trainer.accelerator, etc.).

        Returns:
            Dictionary of metrics to log (can be empty).
        """
        return {}

    def on_test_dataset_start(self, trainer: "Trainer", dataset_name: str):
        """Called at the start of testing a dataset.

        Args:
            trainer: Trainer instance.
            dataset_name: Name of the dataset.
        """
        pass

    def on_test_batch(
        self, batch: dict, predictions: dict, output_dir: str, batch_idx: int, trainer: "Trainer", **kwargs
    ) -> dict:
        """Called after processing a test batch.

        Args:
            batch: Ground truth batch.
            predictions: Model predictions.
            output_dir: Directory to save outputs.
            batch_idx: Index of the batch.
            trainer: Trainer instance (access via trainer.model, trainer.accelerator, etc.).
            **kwargs: Additional arguments (e.g., global_step, dataset_name, trackers).

        Returns:
            Dictionary of metrics to log (can be empty).
        """
        return {}

    def on_test_dataset_end(self, trainer: "Trainer", output_dir: str) -> Optional[dict]:
        """Called at the end of testing a dataset.

        Args:
            trainer: Trainer instance.
            output_dir: Directory where outputs were saved.

        Returns:
            Optional dictionary of aggregated metrics.
        """
        return None
