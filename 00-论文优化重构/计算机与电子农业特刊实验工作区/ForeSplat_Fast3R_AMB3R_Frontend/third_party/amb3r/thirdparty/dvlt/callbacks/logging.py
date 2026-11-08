# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging callbacks for training metrics."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import wandb
from accelerate.logging import get_logger

from dvlt.callbacks.base import Callback, CallbackConfig


if TYPE_CHECKING:
    from dvlt.engine.trainer import Trainer


logger = get_logger(__name__)


@dataclass
class ParameterLoggingCallbackConfig(CallbackConfig):
    """Parameter logging callback configuration."""

    _target_: str = "dvlt.callbacks.logging.ParameterLoggingCallback"
    log_params: Optional[List[str]] = None
    log_every_n_steps: int = 50


class ParameterLoggingCallback(Callback):
    """Callback for logging parameter and gradient statistics.

    This replaces the parameter logging functionality that was previously
    in Module._log_train(). It's now a configurable callback that can log:
    - All trainable parameters
    - Specific modules/parameters
    - Parameter histograms and gradient histograms to wandb/tensorboard
    """

    def __init__(
        self,
        log_params: Optional[List[str]] = None,
        log_every_n_steps: int = 50,
    ):
        """Initialize parameter logging callback.

        Args:
            log_params: Parameter logging configuration:
                - None (default): No parameter logging
                - [] (empty list): Log all trainable parameters
                - ['module1', 'param.weight']: Log specific modules/parameters only
            log_every_n_steps: Log every N steps.
        """
        self.log_params = log_params
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch(
        self,
        batch: dict,
        predictions: dict,
        step: int,
        trainer: "Trainer",
    ) -> dict:
        """Log parameter and gradient statistics.

        Args:
            batch: Training batch.
            predictions: Model predictions.
            step: Current training step.
            trainer: Trainer instance.

        Returns:
            Empty dict.
        """
        if self.log_params is None:
            return {}

        if step % self.log_every_n_steps != 0 and step != 0:
            return {}

        # Get the underlying model, unwrapping DDP if necessary
        model = self._unwrap_model(trainer.model.model)

        if self.log_params == []:
            # Empty list means log all trainable parameters
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                self._log_parameter(name, param, step, trainer.accelerator.trackers)
        else:
            # Log only specified parameters/modules
            for param_path in self.log_params:
                try:
                    # Navigate to the parameter/module using the path
                    target = model
                    for attr in param_path.split("."):
                        target = getattr(target, attr)

                    # If it's a module, log all its parameters
                    if hasattr(target, "named_parameters"):
                        for name, param in target.named_parameters():
                            if not param.requires_grad:
                                continue
                            full_name = f"{param_path}.{name}" if name else param_path
                            self._log_parameter(full_name, param, step, trainer.accelerator.trackers)
                    # If it's a single parameter, log it
                    elif hasattr(target, "requires_grad") and target.requires_grad:
                        self._log_parameter(param_path, target, step, trainer.accelerator.trackers)
                    else:
                        logger.warning(f"Cannot log {param_path}: not a trainable parameter or module")

                except AttributeError as e:
                    logger.warning(f"Cannot log {param_path}: {e}")

        return {}

    def _unwrap_model(self, model):
        """Get the underlying model, unwrapping DDP or other wrappers if necessary."""
        # Handle DistributedDataParallel
        if hasattr(model, "module"):
            model = model.module

        # Handle other potential wrappers that might have a 'module' attribute
        # Keep unwrapping until we reach the base model
        while hasattr(model, "module") and hasattr(model.module, "__class__"):
            model = model.module

        return model

    def _log_parameter(self, name: str, param, global_step: int, trackers):
        """Helper method to log statistics for a single parameter."""
        for tracker in trackers:
            if tracker.name == "wandb":
                tracker.log(
                    {f"param_hist/{name}": wandb.Histogram(param.data.detach().cpu().float())},
                    step=global_step,
                )
            elif tracker.name == "tensorboard":
                tracker.log(
                    {f"param_hist/{name}": param.data.detach().cpu()},
                    step=global_step,
                )

            if param.grad is not None:
                if tracker.name == "wandb":
                    tracker.log(
                        {f"grad_hist/{name}": wandb.Histogram(param.grad.detach().cpu().float())},
                        step=global_step,
                    )
                elif tracker.name == "tensorboard":
                    tracker.log(
                        {f"grad_hist/{name}": param.grad.detach().cpu()},
                        step=global_step,
                    )
