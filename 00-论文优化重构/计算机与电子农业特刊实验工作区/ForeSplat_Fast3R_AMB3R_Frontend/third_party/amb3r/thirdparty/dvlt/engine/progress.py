# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""A tqdm-like progress bar with the option to sequentially print to the console without removing the previous line."""

import time

from accelerate.logging import get_logger
from tqdm.auto import tqdm


logger = get_logger(__name__)


class ProgressBar:
    """Custom progress bar that provides both tqdm behavior and sequential printing.

    This class wraps tqdm to provide:
    1. Standard tqdm progress bar (default)
    2. Sequential printing mode that doesn't overwrite previous lines
    3. Support for postfix values
    4. Integration with accelerate logger
    """

    def __init__(self, *args, use_tqdm=True, print_step_interval=1, **kwargs):
        """Initialize the progress bar.

        Args:
            *args: Arguments to pass to tqdm
            use_tqdm: If True, behaves like standard tqdm. If False, uses sequential printing
            print_step_interval: Print a new line every n steps (only used when use_tqdm=False)
            **kwargs: Keyword arguments to pass to tqdm
        """
        # Store configuration
        self._use_tqdm = use_tqdm
        self._print_interval = print_step_interval
        self._steps_since_print = 0
        self._postfix_dict = {}

        # For tracking averages of numeric postfix values
        self._postfix_accumulators = {}
        self._postfix_counts = {}

        # Store initial position if provided
        self._initial_position = kwargs.get("initial", 0)
        self._position = self._initial_position

        # For independent time tracking in sequential mode
        self._start_time = time.time()
        self._last_print_time = self._start_time

        # Track steps processed since timer started (for accurate rate calculation)
        self._steps_processed = 0

        # Create tqdm instance for tqdm mode
        # If in sequential mode, disable the tqdm display
        if not self._use_tqdm:
            kwargs["disable"] = True

        self._tqdm = tqdm(*args, **kwargs)

        # If tqdm was initialized with a non-zero position, update our tracker
        if hasattr(self._tqdm, "n") and self._tqdm.n > 0:
            self._position = self._tqdm.n

        # Print initial state if in sequential mode with non-zero start
        if not self._use_tqdm and self._position > 0:
            self._do_sequential_print()

    def update(self, n=1):
        """Update the progress bar.

        Args:
            n: Number of steps to increment

        Returns:
            int: The new position
        """
        # Update our position counter
        self._position += n

        # Track steps processed since timer started
        self._steps_processed += n

        # Update the tqdm instance
        self._tqdm.update(n)

        # Handle sequential printing
        if not self._use_tqdm:
            self._steps_since_print += n
            if self._steps_since_print >= self._print_interval:
                self._do_sequential_print()
                self._steps_since_print = 0
                # Reset accumulators after printing
                self._postfix_accumulators = {}
                self._postfix_counts = {}
                # Update last print time
                self._last_print_time = time.time()

        return self._position

    def _do_sequential_print(self):
        """Print a progress line without overwriting the previous one."""
        # Calculate percentage
        percentage = 100 * (self._position / self.total if self.total else 0)

        # Create progress bar visual
        bar_length = 20
        filled_length = int(bar_length * self._position / (self.total or 1))
        bar = "█" * filled_length + " " * (bar_length - filled_length)

        # Format the progress string like tqdm
        if self.total:
            progress_str = f"{percentage:3.0f}%|{bar}| {self._position}/{self.total}"
        else:
            progress_str = f"{self._position}"

        # Add timing info using our own timer instead of relying on tqdm
        current_time = time.time()
        elapsed = current_time - self._start_time

        if elapsed > 0:
            # Format elapsed time as HH:MM:SS
            elapsed_str = self._tqdm.format_interval(elapsed)

            # Calculate remaining time
            if self._steps_processed > 0 and self.total:
                # Calculate remaining steps from current position
                remaining_steps = self.total - self._position
                # Calculate rate based on actual steps processed since timer started
                steps_per_second = self._steps_processed / elapsed
                remaining = remaining_steps / steps_per_second if steps_per_second > 0 else float("inf")
                remaining_str = self._tqdm.format_interval(remaining)
            else:
                remaining_str = "?"

            # Calculate iterations per second based on actual steps processed
            rate = self._steps_processed / elapsed if self._steps_processed > 0 else 0

            # Format like tqdm: "00:01<00:00, 19.90it/s"
            progress_str += f" [{elapsed_str}<{remaining_str}, {rate:.2f}it/s]"

        # Add postfix if any exists
        if self._postfix_dict or self._postfix_accumulators:
            postfix_str = self._format_postfix()
            progress_str = f"{progress_str} {postfix_str}"

        # Log the message
        logger.info(progress_str)

    def _format_postfix(self):
        """Format the postfix dictionary in the same way tqdm does.

        For numeric values, displays the average over the last print_step_interval steps.
        """
        formatted_items = []
        for key in self._postfix_dict.keys():
            # If we have accumulated values for numeric metrics, use the average
            if key in self._postfix_accumulators and self._postfix_counts.get(key, 0) > 0:
                avg_value = self._postfix_accumulators[key] / self._postfix_counts[key]
                formatted_items.append(f"{key}={avg_value:.4g}")
            else:
                value = self._postfix_dict[key]
                if isinstance(value, float):
                    formatted_items.append(f"{key}={value:.4g}")
                else:
                    formatted_items.append(f"{key}={value}")

        return ", ".join(formatted_items)

    def set_postfix(self, ordered_dict=None, **kwargs):
        """Set postfix values to display after the progress bar."""
        # Update the tqdm instance
        self._tqdm.set_postfix(ordered_dict, **kwargs)

        # Update our internal postfix dictionary
        if ordered_dict:
            self._postfix_dict.update(ordered_dict)
            # Accumulate numeric values for averaging
            self._accumulate_numeric_values(ordered_dict)

        if kwargs:
            self._postfix_dict.update(kwargs)
            # Accumulate numeric values for averaging
            self._accumulate_numeric_values(kwargs)

        return self

    def _accumulate_numeric_values(self, values_dict):
        """Accumulate numeric values for averaging at print time."""
        for key, value in values_dict.items():
            if isinstance(value, (int, float)):
                if key not in self._postfix_accumulators:
                    self._postfix_accumulators[key] = 0.0
                    self._postfix_counts[key] = 0

                self._postfix_accumulators[key] += value
                self._postfix_counts[key] += 1

    def close(self):
        """Close the progress bar."""
        # Print final state if in sequential mode
        try:
            if not self._use_tqdm and self._steps_since_print > 0:
                self._do_sequential_print()
        except Exception:
            pass

        # Close the tqdm instance
        self._tqdm.close()

    # Forward properties to tqdm
    @property
    def n(self):
        """Current position."""
        return self._position

    @property
    def total(self):
        """Total steps."""
        return self._tqdm.total if hasattr(self._tqdm, "total") else None

    @property
    def desc(self):
        """Description."""
        return self._tqdm.desc if hasattr(self._tqdm, "desc") else None

    # Compatibility properties
    @property
    def sequential_print_enabled(self):
        """Whether sequential printing is enabled."""
        return not self._use_tqdm

    @property
    def print_step_interval(self):
        """Get the print step interval."""
        return self._print_interval

    @property
    def steps_since_last_print(self):
        """Get steps since last print."""
        return self._steps_since_print
