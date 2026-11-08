# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Tensor utilities."""

from typing import List, Optional

import torch


def pad_tensor_list(
    tensor_list: List[torch.Tensor],
    max_size: Optional[int | tuple[int, ...]] = None,
    fill_value: float = 0.0,
    dim: int | tuple[int, ...] = 0,
) -> torch.Tensor:
    """Pad a list of tensors so they can be stacked.

    By default the first dimension (``dim=0``) is padded to the *longest* length in the list.  You can
    provide ``dim`` as an ``int`` to pad along a single dimension, or as a ``tuple`` to pad multiple
    dimensions (e.g. ``dim=(1, 2)`` to pad both height and width for images).  ``max_size`` can be a
    single ``int`` (for the single-dim case) or a tuple with the same length as ``dim`` to explicitly
    set the target size(s); otherwise the maximum size in the list is used.
    """

    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)

    if max_size is not None:
        if isinstance(max_size, int):
            max_sizes = {dims[0]: max_size}
        else:
            assert len(max_size) == len(dims), "max_size tuple must match dim tuple length"
            max_sizes = {d: s for d, s in zip(dims, max_size)}
    else:
        # compute maxima automatically
        max_sizes = {d: max(t.shape[d] for t in tensor_list) for d in dims}

    # Build output tensor with padded shape
    padded_tensors: List[torch.Tensor] = []
    for tensor in tensor_list:
        target_shape = list(tensor.shape)
        for d in dims:
            target_shape[d] = max_sizes[d]

        # Convert fill_value for boolean tensors
        fv = fill_value
        if tensor.dtype == torch.bool:
            fv = bool(fill_value)
        padded = tensor.new_full(tuple(target_shape), fv)

        # construct slicing indices
        slices = [slice(None)] * tensor.ndim
        for d in dims:
            slices[d] = slice(0, tensor.shape[d])

        padded[tuple(slices)] = tensor
        padded_tensors.append(padded)

    return torch.stack(padded_tensors).to(memory_format=torch.contiguous_format)


def to_device(data, device, dtype=None, non_blocking=False):
    """Move data to device.

    Args:
        data: The data to move to device. Can be a tensor, dict, list, or tuple.
        device: The device to move the data to.
        dtype: The dtype to move the data to.
        non_blocking: If True, try to perform the transfer asynchronously.

    Returns:
        The data moved to the correct device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, dtype=dtype, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: to_device(v, device, dtype=dtype, non_blocking=non_blocking) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device, dtype=dtype, non_blocking=non_blocking) for item in data)
    return data


def verify_type(data, name, path="", expected_type=torch.float32):
    """Recursively verify that tensor entries are of the expected type."""
    if isinstance(data, dict):
        for key, value in data.items():
            verify_type(value, name, f"{path}.{key}" if path else key, expected_type)
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            verify_type(item, name, f"{path}[{i}]", expected_type)
    elif isinstance(data, torch.Tensor) and data.dtype.is_floating_point:
        if data.dtype != expected_type:
            raise ValueError(f"{name} tensor at {path} has dtype {data.dtype}, expected {expected_type}")
