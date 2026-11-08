# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for metrics."""

import math
from typing import Dict, List, Optional, Sequence

import torch
from rich.table import Table


def average_metrics_list(metrics_dict_list: List[Dict[str, float]], get_std: bool = False) -> Dict[str, float]:
    """Given a list of of dictionaries with metrics, return avg and optionally stddev."""
    metrics_dict = {}
    for key in metrics_dict_list[0].keys():
        metric_values = torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
        metric_values = metric_values[~torch.isnan(metric_values)]
        if metric_values.numel() == 0:
            continue
        if get_std:
            key_std, key_mean = torch.std_mean(metric_values)
            metrics_dict[key] = float(key_mean)
            metrics_dict[f"{key}_std"] = float(key_std)
        else:
            metrics_dict[key] = float(metric_values.mean())
    return metrics_dict


def build_aggregated_metrics_table(
    metrics: Dict[str, float],
    title: Optional[str] = None,
    precision: int = 4,
    std_suffix: str = "_std",
    key_order: Optional[Sequence[str]] = None,
    key_alias: Optional[Dict[str, str]] = None,
) -> Table:
    """
    Create a compact table from a single aggregated metrics dict.

    - Rows correspond to metric names; columns show the value and, if present, the std.

    Args:
        metrics: Mapping from metric name to value. If standard deviations are
            available, include entries with keys "{name}{std_suffix}".
        title: Optional table title.
        precision: Number of decimal places for float formatting.
        std_suffix: Suffix used to identify std keys (default: "_std").
        key_order: Optional explicit ordering of metric base names (without suffix).
        key_alias: Optional mapping to rename metric base names for display.

    Returns:
        A rich Table instance.
    """

    def _fmt(v: float) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return "-"
            return f"{v:.{precision}f}"
        try:
            vf = float(v)
            if math.isnan(vf) or math.isinf(vf):
                return "-"
            return f"{vf:.{precision}f}"
        except Exception:
            return str(v)

    # Determine base keys (exclude std keys)
    base_keys_set = set()
    for k in metrics.keys():
        if k.endswith(std_suffix):
            base_keys_set.add(k[: -len(std_suffix)])
        else:
            base_keys_set.add(k)

    # Apply explicit order if provided, otherwise preserve insertion order
    if key_order is not None:
        base_keys = [k for k in key_order if k in base_keys_set]
        # Append any remaining keys not covered by key_order in their original order
        for k in metrics.keys():
            base = k[: -len(std_suffix)] if k.endswith(std_suffix) else k
            if base in base_keys_set and base not in base_keys:
                base_keys.append(base)
    else:
        seen = set()
        base_keys = []
        for k in metrics.keys():
            base = k[: -len(std_suffix)] if k.endswith(std_suffix) else k
            if base not in seen:
                seen.add(base)
                base_keys.append(base)

    # Build table
    table = Table(title=title or "Aggregated Metrics")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)

    # Decide whether to include a std column
    has_any_std = any(f"{k}{std_suffix}" in metrics for k in base_keys)
    table.add_column("Value", justify="right", style="yellow")
    if has_any_std:
        table.add_column("Std", justify="right", style="magenta")

    # Populate rows
    for key in base_keys:
        display_name = key_alias.get(key, key) if key_alias is not None else key
        mean_v = metrics.get(key)
        std_v = metrics.get(f"{key}{std_suffix}") if has_any_std else None
        row = [str(display_name), _fmt(mean_v)]
        if has_any_std:
            row.append(_fmt(std_v))
        table.add_row(*row)

    return table
