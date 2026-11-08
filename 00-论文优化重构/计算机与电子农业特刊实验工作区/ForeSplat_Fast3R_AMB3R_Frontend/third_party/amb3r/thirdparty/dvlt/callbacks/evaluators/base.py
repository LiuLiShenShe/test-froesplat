# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes for evaluators with shared metric aggregation logic."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch import Tensor

from dvlt.callbacks.base import Callback
from dvlt.callbacks.util import align_predictions_to_gt, index_batch_and_predictions, scale_batch_fields
from dvlt.common.constants import DataField
from dvlt.metric.util import average_metrics_list, build_aggregated_metrics_table
from dvlt.util.console import render_table_to_text


if TYPE_CHECKING:
    from dvlt.engine.trainer import Trainer


logger = get_logger(__name__)


def unbatch_metrics(metrics: Dict[str, Tensor]) -> List[Dict[str, float]]:
    """Convert a dict of batched tensors to a list of dicts with scalar values.

    Args:
        metrics: Dictionary mapping metric names to tensors of shape [N] or scalar.

    Returns:
        List of N dictionaries, each mapping metric names to scalar float values.
    """
    first_tensor = next(iter(metrics.values()))
    batch_size = first_tensor.numel()

    return [{k: v.flatten()[i].item() for k, v in metrics.items()} for i in range(batch_size)]


class Evaluator(Callback):
    """Base class for evaluators with common metric aggregation logic.

    Evaluators are specialized callbacks that compute metrics from model
    predictions and ground truth. They handle:
      - Batch preprocessing (indexing, scaling, alignment)
      - Per-batch metric computation
      - Metric aggregation across all batches
      - Pretty-printing results

    Subclasses should override:
      - `_compute_batch_metrics()` to compute metrics for a single batch
      - `_get_table_title()` to return title for aggregated metrics table
    """

    def __init__(
        self,
        *args,
        preprocess_batch: bool = True,
        scale_gt: bool = True,
        align_predictions: bool = True,
        outlier_rejection_iters: int = 1,
        outlier_rejection_percentile: float = 95.0,
        **kwargs,
    ):
        """Initialize evaluator.

        Args:
            preprocess_batch: Whether to automatically preprocess batches (index + scale + align).
            scale_gt: Whether to scale ground truth fields to original scale.
            align_predictions: Whether to apply Sim3 alignment to predictions.
            outlier_rejection_iters: Iterative Umeyama refinement rounds (discard top-percentile
                residual points after each solve). 0 disables.
            outlier_rejection_percentile: Residual percentile cutoff per rejection round.
        """
        super().__init__(*args, **kwargs)
        self.preprocess_batch = preprocess_batch
        self.scale_gt = scale_gt
        self.align_predictions = align_predictions
        self.outlier_rejection_iters = outlier_rejection_iters
        self.outlier_rejection_percentile = outlier_rejection_percentile
        self.metrics_list: List[Dict] = []
        self.seq_names_list: List[str] = []
        self.accelerator: Optional[Accelerator] = None

    def on_test_dataset_start(self, trainer: "Trainer", dataset_name: str):
        """Initialize metrics list for the dataset."""
        self.metrics_list = []
        self.seq_names_list = []

    @staticmethod
    def _extract_seq_name(batch: dict, batch_idx: int) -> str:
        """Extract a single sequence name from a (pre-preprocessed) batch.

        The raw batch from the dataloader carries ``SEQ_NAME`` as a list with
        one entry per batch element. Fallback to a deterministic placeholder
        if the field is missing.
        """
        raw = batch.get(DataField.SEQ_NAME, None)
        if raw is None:
            return f"batch_{batch_idx:05d}"
        if isinstance(raw, (list, tuple)):
            return str(raw[0]) if len(raw) > 0 else f"batch_{batch_idx:05d}"
        return str(raw)

    def _preprocess_batch(self, batch: dict, predictions: dict) -> tuple[dict, dict]:
        """Preprocess batch and predictions.

        Args:
            batch: Ground truth batch.
            predictions: Model predictions.

        Returns:
            Tuple of (preprocessed batch, preprocessed predictions).
        """
        # Get eval indices if available
        eval_indices = batch.get("eval_indices", None)

        with torch.amp.autocast("cuda", enabled=False):
            # 1) Indexing (batch and predictions), optional sequence sub-sampling
            batch, predictions = index_batch_and_predictions(
                batch, predictions, batch_idx=0, seq_idxs=eval_indices, inplace=False
            )

            # 2) Scale GT fields to original scale
            if self.scale_gt:
                batch = scale_batch_fields(batch, inplace=False)

            # 3) Sim3 alignment
            if self.align_predictions:
                predictions = align_predictions_to_gt(
                    batch,
                    predictions,
                    inplace=False,
                    outlier_rejection_iters=self.outlier_rejection_iters,
                    outlier_rejection_percentile=self.outlier_rejection_percentile,
                )

        return batch, predictions

    def on_test_batch(
        self, batch: dict, predictions: dict, output_dir: str, batch_idx: int, trainer: "Trainer", **kwargs
    ) -> dict:
        """Process a batch and compute metrics.

        Args:
            batch: Ground truth batch.
            predictions: Model predictions.
            output_dir: Directory to save outputs.
            batch_idx: Index of the batch.
            trainer: Trainer instance.
            **kwargs: Additional arguments (e.g., global_step, dataset_name).

        Returns:
            Dictionary of metrics for this batch.
        """
        # Extract sequence name from the raw (non-preprocessed) batch so that
        # list-typed SEQ_NAME entries are handled correctly.
        seq_name = self._extract_seq_name(batch, batch_idx)

        # Preprocess batch if enabled
        if self.preprocess_batch:
            batch, predictions = self._preprocess_batch(batch, predictions)

        # Compute metrics for this batch (returns dict[str, Tensor], NaN for missing)
        with torch.amp.autocast("cuda", enabled=False):
            metrics = self._compute_batch_metrics(batch, predictions, output_dir, batch_idx)

        # Check if distributed evaluation is enabled
        distributed_eval = getattr(trainer.data_module, "distributed_eval", True)

        if distributed_eval:
            # Distributed mode: gather metrics from all ranks
            # Normalize all metrics to same dtype (float32) and device (CUDA)
            # This is critical for gather_for_metrics to use consistent NCCL operations across ranks
            device = trainer.accelerator.device
            metrics = {k: v.float().to(device) for k, v in metrics.items()}

            # Synchronize all ranks before gathering
            trainer.accelerator.wait_for_everyone()

            # Gather metrics across all processes (uses NCCL for tensors, handles padding dedup)
            gathered = trainer.accelerator.gather_for_metrics(metrics)

            # Convert batched tensors to list of dicts with scalar values
            gathered_list = unbatch_metrics(gathered)
            self.metrics_list.extend(gathered_list)

            # Gather per-rank sequence names (strings). ``gather_for_metrics``
            # falls back to ``gather_object`` when the input is not a tensor
            # structure; we take the first ``len(gathered_list)`` entries to
            # stay aligned with the (potentially padding-deduped) metrics.
            gathered_names = trainer.accelerator.gather_for_metrics([seq_name])
            if not isinstance(gathered_names, list):
                gathered_names = list(gathered_names)
            self.seq_names_list.extend(str(n) for n in gathered_names[: len(gathered_list)])

            return gathered_list[0]  # return only one rank's metrics
        # Non-distributed mode: only rank 0 collects metrics (no NCCL operations)
        # All GPUs process all samples independently, only rank 0 logs results
        elif trainer.accelerator.is_main_process:
            # Convert tensors to scalar floats for aggregation
            metrics_scalar = {k: v.float().item() for k, v in metrics.items()}
            self.metrics_list.append(metrics_scalar)
            self.seq_names_list.append(seq_name)
            return metrics_scalar
        else:
            # Non-main ranks return empty metrics (they still process for timing consistency)
            return {k: float("nan") for k in metrics.keys()}

    def on_test_dataset_end(self, trainer: "Trainer", output_dir: str) -> Optional[dict]:
        """Aggregate metrics and print summary.

        Args:
            trainer: Trainer instance.
            output_dir: Directory where outputs were saved.

        Returns:
            Dictionary of aggregated metrics. When per-sequence metrics are
            available, a ``"per_sequence"`` entry is also included, mapping
            each sequence name to its (scalar) metric dict.
        """
        metrics_agg = {}
        if trainer.accelerator.is_main_process and len(self.metrics_list) > 0:
            logger.info(f"Aggregate {self._get_table_title()} across {len(self.metrics_list)} samples")
            metrics_agg = average_metrics_list(self.metrics_list, get_std=True)

            if metrics_agg:
                table = build_aggregated_metrics_table(
                    metrics=metrics_agg,
                    title=self._get_table_title(),
                )
                logger.info(f"\n{render_table_to_text(table)}")

            # Build per-sequence metrics dict. If sequence names are missing
            # or misaligned (e.g. non-distributed partial state), fall back to
            # synthetic ``seq_{idx}`` names. Duplicate names keep the last
            # occurrence, matching natural dict semantics.
            per_sequence: Dict[str, Dict[str, float]] = {}
            n_metrics = len(self.metrics_list)
            n_names = len(self.seq_names_list)
            if n_names != n_metrics:
                logger.warning(
                    f"Evaluator seq-name count ({n_names}) != metrics count ({n_metrics}); "
                    f"falling back to synthetic sequence names where missing."
                )
            for i, metrics in enumerate(self.metrics_list):
                name = self.seq_names_list[i] if i < n_names else f"seq_{i:05d}"
                per_sequence[name] = {k: float(v) for k, v in metrics.items()}

            if per_sequence:
                metrics_agg["per_sequence"] = per_sequence

        # Cleanup
        del self.metrics_list
        del self.seq_names_list
        self.metrics_list = []
        self.seq_names_list = []

        return metrics_agg

    @abstractmethod
    def _compute_batch_metrics(
        self, batch: dict, predictions: dict, output_dir: str, batch_idx: int
    ) -> Dict[str, Tensor]:
        """Compute metrics for a single batch.

        IMPORTANT: Must return ALL metric keys with NaN tensor for missing values.
        This ensures all ranks have identical dict structure for distributed gathering.

        Args:
            batch: Preprocessed ground truth batch.
            predictions: Model predictions.
            output_dir: Directory to save outputs.
            batch_idx: Index of the batch.

        Returns:
            Dictionary mapping metric names to scalar tensors. Use torch.tensor(float('nan'))
            for metrics that cannot be computed (e.g., due to missing GT).
        """
        raise NotImplementedError

    @abstractmethod
    def _get_table_title(self) -> str:
        """Return the title for the aggregated metrics table."""
        raise NotImplementedError
