# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Any, List

import torch

from dvlt.common.tensor import pad_tensor_list


# Padding behaviour per DataField.
# Each entry can be either a scalar fill value (pad along dim 0) or
# a dict specifying the fill value and the dimension(s) along which to pad.
PADDING_STRATEGIES: dict[str, object] = {}


def _collate_tensor_lists(
    tensor_lists: List[List[torch.Tensor]],
    fill_value: float = 0.0,
    dim: int | tuple[int, ...] = 0,
) -> torch.Tensor:
    """Pad and stack a list of list-tensors.

    Each element in *tensor_lists* is itself a list (length S) of tensors that share the
    same shape except possibly along *dim*.
    """
    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)

    # compute maxima
    max_sizes = {d: 0 for d in dims}
    for lst in tensor_lists:
        for t in lst:
            for d in dims:
                max_sizes[d] = max(max_sizes[d], t.shape[d])

    max_size_arg = max_sizes[dims[0]] if len(dims) == 1 else tuple(max_sizes[d] for d in dims)

    padded_lists = [pad_tensor_list(x, max_size=max_size_arg, fill_value=fill_value, dim=dim) for x in tensor_lists]
    return torch.stack(padded_lists).to(memory_format=torch.contiguous_format)


def default_collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Data-Loder collate function that applies the padding rules above."""
    result: dict[str, Any] = {}

    for key in batch[0].keys():
        values = [sample[key] for sample in batch]

        if isinstance(values[0], torch.Tensor):
            strat = PADDING_STRATEGIES.get(key, None)
            if strat is not None:
                if not isinstance(strat, dict):
                    strat = {"fill_value": strat, "dim": 0}
                result[key] = pad_tensor_list(
                    values,
                    fill_value=strat.get("fill_value", 0),
                    dim=strat.get("dim", 0),
                )
            else:
                result[key] = torch.stack(values).to(memory_format=torch.contiguous_format)
        elif isinstance(values[0], dict):
            result[key] = default_collate_fn(values)
        elif isinstance(values[0], list) and values[0] and isinstance(values[0][0], torch.Tensor):
            strat = PADDING_STRATEGIES.get(key, None)
            if isinstance(strat, dict):
                fill_value = strat.get("fill_value", 0)
                dim_val = strat.get("dim", 0)
            elif strat is not None:
                fill_value = strat
                dim_val = 0
            else:
                fill_value = 0
                dim_val = 0
            result[key] = _collate_tensor_lists(values, fill_value, dim=dim_val)
        else:
            result[key] = values

    return result
