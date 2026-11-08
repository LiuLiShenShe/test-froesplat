# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MultiTaskLoss for combining multiple loss functions."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn

from dvlt.common.constants import DataField, PredictionField
from dvlt.config.schema import LossConfig
from dvlt.model_components.loss.util import compute_normalization_scale


# Keys used to store the normalization scale factors
PRED_NORMALIZATION_SCALE_KEY = "_pred_normalization_scale"  # stored in predictions dict
BATCH_NORMALIZATION_SCALE_KEY = "_batch_normalization_scale"  # stored in batch dict


def _resolve_loss_config(entry: Union[LossConfig, Mapping[str, Any]]) -> LossConfig:
    """Normalize a loss config entry to a LossConfig.

    Accepts either a LossConfig dataclass (from Python defaults) or a plain
    dict/DictConfig (from YAML) with a string ``loss_class`` dotted path.
    """
    if isinstance(entry, LossConfig):
        return entry

    loss_class = entry["loss_class"]
    if isinstance(loss_class, str):
        module_path, class_name = loss_class.rsplit(".", 1)
        loss_class = getattr(importlib.import_module(module_path), class_name)

    kwargs = dict(entry.get("kwargs") or {})
    return LossConfig(
        loss_class=loss_class,
        weight=float(entry.get("weight", 1.0)),
        enabled=bool(entry.get("enabled", True)),
        kwargs=kwargs,
    )


class MultiTaskLoss(nn.Module):
    """Combines multiple loss functions with configurable weights.

    Each loss module should implement:
        - forward(predictions, batch) -> Tuple[Tensor, Dict[str, Tensor]]
          where the returned Tensor has shape [B] (per-sample losses)
        - check_inputs(predictions, batch) -> bool (optional)

    Loss configs can be LossConfig dataclasses (Python) or plain dicts with
    a string ``loss_class`` dotted path (YAML / Hydra overrides).
    """

    def __init__(
        self,
        losses: Dict[str, Union[LossConfig, Mapping[str, Any]]],
        skip_missing: bool = True,
        normalize: bool = True,
    ):
        """Initialize MultiTaskLoss.

        Args:
            losses: Dictionary mapping loss names to their configurations.
                Each value can be a LossConfig or a dict with keys
                ``loss_class`` (str or type), ``weight``, ``enabled``, ``kwargs``.
            skip_missing: If True, skip losses when required inputs are missing.
                         If False, raise an error when inputs are missing.
            normalize: If True, compute normalization scale factor from world points
                      and store in batch for individual losses to apply.
        """
        super().__init__()
        self.skip_missing = skip_missing
        self.normalize = normalize

        # Store configs and instantiate losses
        self._configs: Dict[str, LossConfig] = {}
        self._losses = nn.ModuleDict()
        self._weights: Dict[str, float] = {}
        self._enabled: Dict[str, bool] = {}

        for name, entry in losses.items():
            config = _resolve_loss_config(entry)
            self._configs[name] = config
            self._losses[name] = config.loss_class(**config.kwargs)
            self._weights[name] = config.weight
            self._enabled[name] = config.enabled

    def forward(
        self,
        predictions: Dict[str, Any],
        batch: Dict[str, Any],
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute all enabled losses and return weighted sum.

        Args:
            predictions: Dictionary of model predictions.
            batch: Dictionary of ground truth data.
            sample_weights: Optional per-sample weights of shape [B]. Used for
                diffusion models to apply time-dependent loss weighting before
                batch reduction.

        Returns:
            Tuple of (total_loss, pbar_dict, tracker_dict):
                - total_loss: Weighted sum of all losses (scalar).
                - pbar_dict: Dictionary of loss values for progress bar.
                - tracker_dict: Dictionary of all loss components for tracking.
        """
        # Compute normalization scale factor once before computing any losses
        if self.normalize:
            predictions, batch = self._compute_normalization_scale(predictions, batch)

        total_loss = None
        pbar_dict: Dict[str, torch.Tensor] = {}
        tracker_dict: Dict[str, torch.Tensor] = {}

        for name, loss_fn in self._losses.items():
            if not self._enabled[name]:
                continue

            # Check if inputs are available (if the loss has check_inputs method)
            if hasattr(loss_fn, "check_inputs"):
                if not loss_fn.check_inputs(predictions, batch):
                    if self.skip_missing:
                        continue
                    raise ValueError(
                        f"Loss '{name}' missing required inputs. "
                        f"Predictions: {list(predictions.keys())}, "
                        f"Batch: {list(batch.keys())}"
                    )

            loss, loss_dict = loss_fn(predictions, batch)  # loss is [B] per-sample
            weight = self._weights[name]
            weighted_loss = loss * weight  # [B]

            # Apply per-sample weights (e.g., diffusion time-dependent weighting)
            if sample_weights is not None:
                weighted_loss = weighted_loss * sample_weights  # [B] * [B] -> [B]

            # Reduce to scalar
            weighted_loss = weighted_loss.mean()

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

            # Add to progress bar dict (use the unweighted per-sample mean)
            pbar_dict[f"loss_{name}"] = loss.mean()

            # Add all components to tracker dict with prefix
            for key, value in loss_dict.items():
                tracker_dict[f"{name}_{key}"] = value

        if total_loss is None:
            raise RuntimeError(
                f"No losses were computed. All losses were either disabled or skipped. "
                f"Enabled losses: {[n for n, e in self._enabled.items() if e]}, "
                f"Prediction keys: {list(predictions.keys())}, "
                f"Batch keys: {list(batch.keys())}"
            )

        return total_loss, pbar_dict, tracker_dict

    def set_weight(self, name: str, weight: float) -> None:
        """Update the weight for a specific loss."""
        if name not in self._weights:
            raise KeyError(f"Loss '{name}' not found. Available: {list(self._weights.keys())}")
        self._weights[name] = weight

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a specific loss."""
        if name not in self._enabled:
            raise KeyError(f"Loss '{name}' not found. Available: {list(self._enabled.keys())}")
        self._enabled[name] = enabled

    def get_weight(self, name: str) -> float:
        """Get the weight for a specific loss."""
        return self._weights[name]

    def is_enabled(self, name: str) -> bool:
        """Check if a specific loss is enabled."""
        return self._enabled[name]

    @property
    def loss_names(self) -> List[str]:
        """Return list of all loss names."""
        return list(self._losses.keys())

    def _compute_normalization_scale(
        self,
        predictions: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute normalization scale factors from world points.

        Computes separate scale factors for predictions and batch (GT).
        Prediction scale stored in predictions dict, batch scale in batch dict.

        Args:
            predictions: Dictionary of model predictions.
            batch: Dictionary of ground truth data.

        Returns:
            Tuple of (updated predictions dict, updated batch dict).
        """
        mask = batch[DataField.POINT_MASKS]
        predictions = dict(predictions)  # Shallow copy to avoid modifying original
        batch = dict(batch)

        # Compute scale from predicted world points
        pred_world_points = predictions.get(PredictionField.WORLD_POINTS)
        if pred_world_points is None:
            raise ValueError("World points not found in predictions")
        predictions[PRED_NORMALIZATION_SCALE_KEY] = compute_normalization_scale(pred_world_points, mask)

        # Compute scale from GT world points
        gt_world_points = batch.get(DataField.WORLD_POINTS)
        if gt_world_points is None:
            raise ValueError("World points not found in batch")
        # Note: This will usually be 1.0 because the GT is normalized in the data pipeline, but we
        # still compute it here to ensure consistency with the prediction scale.
        batch[BATCH_NORMALIZATION_SCALE_KEY] = compute_normalization_scale(gt_world_points, mask)

        return predictions, batch
