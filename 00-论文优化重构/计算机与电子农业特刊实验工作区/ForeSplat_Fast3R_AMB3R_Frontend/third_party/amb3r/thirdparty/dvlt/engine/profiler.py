# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.cuda._memory_viz as viz
from accelerate.logging import get_logger
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.profiler import (
    _ExperimentalConfig as ExperimentalConfig,
)


logger = get_logger(__name__)


class ProfilerMode(str, Enum):
    """Profiler modes."""

    MINIMAL = "minimal"  # Minimal profiling - just data and step timing
    FULL = "full"  # Full timing of high-level operations
    PYTORCH = "pytorch"  # PyTorch profiler with optional memory profiling
    MEMORY = "memory"  # CUDA memory history with stack traces and snapshot dump


class Profiler:
    """Profiler class for memory usage and timing measurements.

    This class provides functionality to profile memory usage and timing of operations
    during training and inference.

    Modes:
    - MINIMAL: Tracks only data loading time and total step time, minimal overhead
    - FULL: Full timing of all operations (forward, backward, optimizer, data loading, etc.)
               and memory usage statistics
    - PYTORCH: PyTorch native profiler for detailed timing and memory statistics,
               with TensorBoard integration for visualization
    - MEMORY: CUDA memory history with stack traces using record_memory_history, dumps a snapshot
              for offline visualization
    """

    def __init__(
        self,
        mode: Union[str, ProfilerMode] = ProfilerMode.MINIMAL,
        output_dir: str = "/tmp",
        # PyTorch profiler parameters
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        experimental_verbose: bool = True,
        # Scheduler parameters (for PyTorch profiler)
        schedule_wait: int = 1,
        schedule_warmup: int = 1,
        schedule_active: int = 5,
        schedule_repeat: int = 1,
        schedule_skip_first: int = 3,
    ):
        """Initialize the profiler.

        Args:
            mode: Profiling mode ('minimal', 'full', 'pytorch')
            output_dir: Directory to save profiler outputs (PyTorch traces).

            # PyTorch profiler options
            record_shapes: Whether to record tensor shapes in PyTorch profiler
            profile_memory: Whether to record memory usage in PyTorch profiler
                           (significantly increases output size)
            with_stack: Whether to record stack traces in PyTorch profiler
                       (significantly increases output size)
            experimental_verbose: Enable experimental verbose tracing when available

            # PyTorch profiler scheduler options
            schedule_wait: Steps to wait before profiling
            schedule_warmup: Steps to warmup before profiling
            schedule_active: Steps to actively profile
            schedule_repeat: Number of profiling cycles to run (0 for unlimited)
            schedule_skip_first: Steps to skip at the beginning
        """
        if isinstance(mode, str):
            self.mode = ProfilerMode(mode.lower())
        else:
            self.mode = mode

        self.output_dir = Path(output_dir)
        self.start_times = {}
        self.pytorch_profiler = None
        self.memory_history_enabled = False
        self.memory_snapshot_path = None

        # Unified step counter for all profiling modes
        self.profiler_step = 0

        # Track timing metrics for the current step
        self.current_step_metrics = {}

        # PyTorch profiler settings
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.experimental_verbose = experimental_verbose

        # Scheduler parameters
        self.schedule_wait = schedule_wait
        self.schedule_warmup = schedule_warmup
        self.schedule_active = schedule_active
        self.schedule_repeat = schedule_repeat
        self.schedule_skip_first = schedule_skip_first

        # Calculate step limit for advanced profiling modes
        self.step_limit = None
        if self.mode in [ProfilerMode.PYTORCH, ProfilerMode.MEMORY]:
            cycle_steps = self.schedule_wait + self.schedule_warmup + self.schedule_active
            if self.schedule_repeat > 0:
                self.step_limit = self.schedule_skip_first + (cycle_steps * self.schedule_repeat)
            else:
                # If repeat is 0, we use a default of 2 cycles for step limiting
                self.step_limit = self.schedule_skip_first + (cycle_steps * 2)

            os.makedirs(self.output_dir, exist_ok=True)

    def start_step(self, global_step: int = 0):
        """Start profiling for a training/inference step.

        Args:
            global_step: Global step number
        """
        # Reset metrics for the new step
        self.current_step_metrics = {}

        # Even in MINIMAL mode, we always track step timing
        self.start_timer("total")

        if self.mode == ProfilerMode.MINIMAL:
            return

        # Start PyTorch profiler if in PYTORCH mode and not already running
        # and only start it once at the beginning of training
        if self.mode == ProfilerMode.PYTORCH and self.pytorch_profiler is None and global_step == 0:
            self._initialize_pytorch_profiler()
        # Start CUDA memory history if in MEMORY mode and not already enabled
        if self.mode == ProfilerMode.MEMORY and not self.memory_history_enabled and global_step == 0:
            self._initialize_memory_profiling()

        # Create a per-step context to organize allocations under the step
        if self.mode in [ProfilerMode.PYTORCH, ProfilerMode.MEMORY]:
            self.step_record_context = record_function(f"train_step_{global_step}")
            self.step_record_context.__enter__()

    def end_step(self, global_step: int = 0):
        """End profiling for a training/inference step and calculate metrics.

        Args:
            global_step: Global step number

        Returns:
            Dict containing profiling metrics for this step
        """
        # Always get step timing, even in MINIMAL mode
        metrics = {}
        total_time = self.stop_timer("total")
        if total_time is not None:
            self.current_step_metrics["total_time"] = total_time

        # Calculate "other" time - time spent in operations not explicitly measured
        self._calculate_other_time()

        # Copy all timing metrics to the returned metrics
        metrics.update(self.current_step_metrics)

        if self.mode == ProfilerMode.MINIMAL:
            return metrics

        # Collect memory stats for FULL and PYTORCH modes
        if torch.cuda.is_available() and self.mode in [ProfilerMode.FULL, ProfilerMode.PYTORCH]:
            memory_stats = self.get_memory_stats()
            metrics.update(memory_stats)
            torch.cuda.reset_peak_memory_stats()

        # Increment the unified step counter for advanced profiling modes
        if self.mode == ProfilerMode.PYTORCH:
            self.profiler_step += 1
        elif self.mode == ProfilerMode.MEMORY:
            self.profiler_step += 1

        # Step PyTorch profiler forward
        if self.mode == ProfilerMode.PYTORCH and self.pytorch_profiler is not None:
            self._step_pytorch_profiler()

        # Finalize memory history and dump snapshot when reaching step limit
        if self.mode == ProfilerMode.MEMORY and self.step_limit is not None and self.profiler_step >= self.step_limit:
            self._finalize_memory_profiling()

        # Close the per-step context if opened
        if hasattr(self, "step_record_context"):
            self.step_record_context.__exit__(None, None, None)
            delattr(self, "step_record_context")

        return metrics

    def _calculate_other_time(self):
        """Calculate the time spent in operations not explicitly measured.

        This calculates 'other_time' as the difference between total step time and
        the sum of all other measured times.

        Note: This is skipped in MINIMAL mode which only tracks data_time and total_time.
        """
        # Skip other_time calculation in MINIMAL mode
        if self.mode == ProfilerMode.MINIMAL:
            return

        if "total_time" not in self.current_step_metrics:
            return

        other_time = self.current_step_metrics["total_time"]

        # Subtract all measured times except total_time
        for key, value in self.current_step_metrics.items():
            if key not in {"total_time", "other_time"} and isinstance(value, (int, float)):
                other_time -= value

        # Only add other_time if it's significant and positive
        if other_time > 0.001:
            self.current_step_metrics["other_time"] = other_time

    def _handle_pytorch_record_context(self, name: str, start: bool = True):
        """Handle context creation and cleanup for PyTorch profiler record_function.

        Args:
            name: Name of the operation being profiled
            start: Whether to start (True) or stop (False) the context
        """
        if self.mode not in [ProfilerMode.PYTORCH, ProfilerMode.MEMORY]:
            return

        context_attr = f"record_context_{name}"
        if start:
            context = record_function(name)
            context.__enter__()
            setattr(self, context_attr, context)
        elif hasattr(self, context_attr):
            context = getattr(self, context_attr)
            context.__exit__(None, None, None)
            delattr(self, context_attr)

    def _time_operation(self, name: str, start: bool = True, record_metric: bool = True):
        """Common handling for starting/stopping operation timing.

        Args:
            name: Name of the operation being timed
            start: Whether to start (True) or stop (False) the timer
            record_metric: Whether to record the metric in current_step_metrics

        Returns:
            Elapsed time in seconds (if stop=True), or None
        """
        # Special case for data_loading, we want it to be stored as "data_time"
        metric_name = "data_time" if name == "data_loading" else f"{name}_time"

        # MINIMAL mode only tracks total and data_loading times
        # FULL mode tracks all operations
        # PyTorch mode also records contexts for all operations
        should_time = False

        # Always time total and data_loading in all modes
        if name in ["total", "data_loading"]:
            should_time = True
        # In FULL and PYTORCH/MEMORY modes, time all operations
        elif self.mode in [ProfilerMode.FULL, ProfilerMode.PYTORCH, ProfilerMode.MEMORY]:
            should_time = True

        if start:
            if should_time:
                self.start_timer(name)
            # For PyTorch profiler and MEMORY mode, use record context for all operations
            if self.mode in [ProfilerMode.PYTORCH, ProfilerMode.MEMORY]:
                self._handle_pytorch_record_context(name, start=True)
            return None
        else:
            elapsed_time = None
            # Get elapsed time if we're using timers
            if should_time:
                elapsed_time = self.stop_timer(name)

                # Store the time in our metrics dictionary
                if record_metric and elapsed_time is not None:
                    self.current_step_metrics[metric_name] = elapsed_time

            # Clean up record context if needed
            if self.mode in [ProfilerMode.PYTORCH, ProfilerMode.MEMORY]:
                self._handle_pytorch_record_context(name, start=False)

            return elapsed_time

    def start_timer(self, name: str):
        """Start a timer.

        Args:
            name: Name of the timer.
        """
        # Always track timing regardless of mode
        self.start_times[name] = time.perf_counter()

    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a timer and return the elapsed time.

        Args:
            name: Name of the timer.

        Returns:
            Elapsed time in seconds, or None if timing is disabled.
        """
        if name not in self.start_times:
            return None

        elapsed = time.perf_counter() - self.start_times[name]
        del self.start_times[name]
        return elapsed

    def start_forward(self):
        """Start timing the forward pass."""
        return self._time_operation("forward", start=True)

    def stop_forward(self) -> Optional[float]:
        """Stop timing the forward pass and return the elapsed time."""
        return self._time_operation("forward", start=False)

    def start_backward(self):
        """Start timing the backward pass."""
        return self._time_operation("backward", start=True)

    def stop_backward(self) -> Optional[float]:
        """Stop timing the backward pass and return the elapsed time."""
        return self._time_operation("backward", start=False)

    def start_optimizer(self):
        """Start timing optimizer operations (grad clipping, opt step, scheduler, zero_grad)."""
        return self._time_operation("optimizer", start=True)

    def stop_optimizer(self) -> Optional[float]:
        """Stop timing optimizer operations and return the elapsed time."""
        return self._time_operation("optimizer", start=False)

    def start_data_loading(self):
        """Start timing data loading."""
        return self._time_operation("data_loading", start=True)

    def stop_data_loading(self) -> Optional[float]:
        """Stop timing data loading and return the elapsed time."""
        return self._time_operation("data_loading", start=False)

    def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        """Get current CUDA memory statistics.

        Returns:
            Dictionary with memory statistics in GB with fixed decimal formatting.
        """
        memory_stats = {}

        # Only collect stats if CUDA is available
        if not torch.cuda.is_available():
            return memory_stats

        # Define metrics to collect
        memory_metrics = [
            # (metric_name, function_to_call, divisor_to_gb)
            ("allocated_gb", torch.cuda.memory_allocated, 1024 * 1024 * 1024),
            ("cached_gb", torch.cuda.memory_reserved, 1024 * 1024 * 1024),
            ("max_allocated_gb", torch.cuda.max_memory_allocated, 1024 * 1024 * 1024),
            ("max_cached_gb", torch.cuda.max_memory_reserved, 1024 * 1024 * 1024),
        ]
        # Collect each metric with error handling
        for metric_name, metric_fn, divisor in memory_metrics:
            value = metric_fn() / divisor
            memory_stats[metric_name] = round(value, 2)

        return memory_stats

    def __enter__(self):
        """Context manager entry."""
        self.start_step()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Make sure to clean up PyTorch profiler if it's still running
        if self.mode == ProfilerMode.PYTORCH and self.pytorch_profiler is not None:
            self.pytorch_profiler.__exit__(None, None, None)
            self.pytorch_profiler = None

        # For MEMORY mode, if still enabled, finalize and dump snapshot
        if self.mode == ProfilerMode.MEMORY and self.memory_history_enabled:
            self._finalize_memory_profiling()

        self.end_step()

    @staticmethod
    def print_viewing_instructions(output_dir: str) -> None:
        """Print instructions for viewing profiler results in TensorBoard.

        Args:
            output_dir: Base directory where profiler data is saved
        """
        tb_dir = os.path.join(output_dir, "tensorboard")
        logger.info("To view profiler results in TensorBoard, run:")
        logger.info(f"Run: tensorboard --logdir={tb_dir}")

    @staticmethod
    def print_memory_viewing_instructions(
        snapshot_path: str, html_path: Optional[str] = None, port: int = 8000
    ) -> None:
        """Print instructions for viewing CUDA memory snapshot locally or from remote.

        Args:
            snapshot_path: Path to the saved CUDA memory snapshot (.pickle)
            html_path: Optional path to the saved HTML report. If None, uses snapshot_path with .html
            port: Port to use for python -m http.server
        """
        html_out = html_path if html_path is not None else os.path.splitext(snapshot_path)[0] + ".html"
        logger.info(
            f"View HTML report at: http://localhost:{port}/{os.path.basename(html_out)} (serve with: python -m http.server {port})"
        )

    def _initialize_pytorch_profiler(self) -> None:
        """Initialize PyTorch profiler."""
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Set up directory for TensorBoard output
        tb_dir = os.path.join(str(self.output_dir), "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)

        # Create a schedule for the profiler
        profiler_schedule = schedule(
            skip_first=self.schedule_skip_first,
            wait=self.schedule_wait,
            warmup=self.schedule_warmup,
            active=self.schedule_active,
            repeat=self.schedule_repeat,
        )

        # Log profiler configuration for debugging
        logger.info(
            f"Starting PyTorch profiler with schedule: "
            f"skip_first={self.schedule_skip_first}, "
            f"wait={self.schedule_wait}, "
            f"warmup={self.schedule_warmup}, "
            f"active={self.schedule_active}, "
            f"repeat={self.schedule_repeat}"
        )
        logger.info(
            f"Profiler settings: activities={activities}, "
            f"record_shapes={self.record_shapes}, "
            f"profile_memory={self.profile_memory}, "
            f"with_stack={self.with_stack}, "
        )
        logger.info(f"Saving profiler data to TensorBoard format: {tb_dir}")

        # Optional experimental verbose config (best-effort, version dependent)
        experimental_config = ExperimentalConfig(verbose=True) if self.experimental_verbose else None

        # Initialize the profiler with the schedule and TensorBoard handler
        profiler_kwargs = dict(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=tensorboard_trace_handler(tb_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        if experimental_config is not None:
            profiler_kwargs["experimental_config"] = experimental_config

        self.pytorch_profiler = profile(**profiler_kwargs)

        # Start the profiler
        self.pytorch_profiler.__enter__()

    def _initialize_memory_profiling(self) -> None:
        """Initialize CUDA memory profiling."""
        if not torch.cuda.is_available():
            logger.error("CUDA is not available; switching MEMORY profiler to FULL mode")
            self.mode = ProfilerMode.FULL
            return

        torch.cuda.memory._record_memory_history(max_entries=100000)  # type: ignore[attr-defined]
        self.memory_history_enabled = True
        logger.info(f"CUDA memory history recording enabled. " f"Snapshots will be saved to: {self.output_dir}")

    def _step_pytorch_profiler(self) -> None:
        """Step the PyTorch profiler and handle cleanup when done."""
        # Step the profiler
        self.pytorch_profiler.step()

        # Check if we've reached the total expected steps
        if self.profiler_step >= self.step_limit:
            self.pytorch_profiler.__exit__(None, None, None)
            self.pytorch_profiler = None
            # Switch to FULL mode after PyTorch profiling is done
            self.mode = ProfilerMode.FULL

    def _finalize_memory_profiling(self) -> None:
        """Dump CUDA memory snapshot and disable memory history."""
        if not self.memory_history_enabled:
            return
        if not torch.cuda.is_available():
            return

        # Determine destination path
        snapshot_file = self.output_dir / "cuda_memory_snapshot.pickle"

        torch.cuda.synchronize()
        torch.cuda.memory_allocated()
        torch.cuda.memory_reserved()
        torch.cuda.memory._dump_snapshot(str(snapshot_file))  # type: ignore[attr-defined]
        self.memory_snapshot_path = str(snapshot_file)
        html_out = os.path.splitext(self.memory_snapshot_path)[0] + ".html"

        # Load snapshot
        with open(self.memory_snapshot_path, "rb") as f:
            snapshot = pickle.load(f)

        # Create HTML report
        with open(html_out, "w") as f:
            f.write(viz.trace_plot(snapshot))

        self.memory_html_path = html_out
        logger.info(f"CUDA memory HTML report saved to {self.memory_html_path}")

        # Always clean up memory history
        torch.cuda.memory._record_memory_history(enabled=False)  # type: ignore[attr-defined]
        self.memory_history_enabled = False
        # Switch to FULL mode after memory profiling is done
        self.mode = ProfilerMode.FULL
