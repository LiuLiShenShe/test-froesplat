#!/usr/bin/env python3
"""Adapter layer: SAM3 geometric-prompt interface with POINT support.

`Sam3Processor.add_geometric_prompt()` only accepts boxes.  The RAP-FSAM3
manuscript (§2.3.4 SB-FBPR) requires distance-transform positive points and
NMS negative points, so this adapter appends point embeddings directly to the
`Prompt` object used by the geometry encoder.

Interface map (SAM3-latest, sam3/model/geometry_encoders.py):

  Prompt.append_points(points, labels, mask=None)
    points : FloatTensor  (N_points, bs, 2)  normalized cx,cy in [0,1]
    labels : LongTensor   (N_points, bs)     1 = positive, 0 = negative
    mask   : BoolTensor   (bs, N_points)     padding mask (default all-visible)

  Prompt.append_boxes(boxes, labels, mask=None)
    boxes  : FloatTensor  (N_boxes, bs, 4)   normalized cx,cy,w,h
    labels : BoolTensor   (N_boxes, bs)      True = positive, False = negative

Both are consumed by `SequenceGeometryEncoder.forward()` ->
`_encode_points()` / `_encode_boxes()` and concatenated into the prompt token
sequence, so points and boxes can be mixed in one forward pass.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch


def normalized_points_tensor(
    points: Sequence[tuple[float, float]],
    device: str,
) -> torch.Tensor:
    """[(x,y) pixel pairs] -> (N, bs=1, 2) normalized [0,1] float tensor."""
    if not points:
        return torch.zeros((0, 1, 2), dtype=torch.float32, device=device)
    arr = torch.tensor(
        [[float(x), float(y)] for x, y in points], dtype=torch.float32, device=device
    )
    return arr.view(-1, 1, 2)


def normalized_boxes_tensor(
    boxes: Sequence[Sequence[float]],
    device: str,
) -> torch.Tensor:
    """[[cx,cy,w,h] normalized] -> (N, bs=1, 4) float tensor."""
    if not boxes:
        return torch.zeros((0, 1, 4), dtype=torch.float32, device=device)
    arr = torch.tensor(
        [[float(v) for v in box] for box in boxes], dtype=torch.float32, device=device
    )
    return arr.view(-1, 1, 4)


def labels_tensor(labels: Sequence[int], device: str) -> torch.Tensor:
    if not labels:
        return torch.zeros((0, 1), dtype=torch.long, device=device)
    arr = torch.tensor([int(v) for v in labels], dtype=torch.long, device=device)
    return arr.view(-1, 1)


def bool_labels_tensor(labels: Sequence[bool], device: str) -> torch.Tensor:
    if not labels:
        return torch.zeros((0, 1), dtype=torch.bool, device=device)
    arr = torch.tensor([bool(v) for v in labels], dtype=torch.bool, device=device)
    return arr.view(-1, 1)


def append_point_prompts(
    state: dict[str, Any],
    positive_points: Sequence[tuple[float, float]],
    negative_points: Sequence[tuple[float, float]],
    device: str,
) -> dict[str, int]:
    """Append positive/negative point prompts onto the live geometric prompt.

    Must be called AFTER `set_text_prompt()` so `state["geometric_prompt"]`
    exists.  Normalized coordinates are (x/W, y/H).
    """
    if "geometric_prompt" not in state:
        raise RuntimeError("geometric_prompt missing: call set_text_prompt() first")
    prompt = state["geometric_prompt"]
    added = {"positive": 0, "negative": 0}
    if positive_points:
        prompt.append_points(
            normalized_points_tensor(positive_points, device),
            labels_tensor([1] * len(positive_points), device),
        )
        added["positive"] = len(positive_points)
    if negative_points:
        prompt.append_points(
            normalized_points_tensor(negative_points, device),
            labels_tensor([0] * len(negative_points), device),
        )
        added["negative"] = len(negative_points)
    return added
