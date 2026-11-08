# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Placeholder reader for dataset targets without a shipped loader."""

from typing import Any, List

from . import DataField
from .base import BaseDataset


class GenericDataset(BaseDataset):
    def __init__(self, target: str | None = None, **kwargs: Any):
        raise NotImplementedError(
            f"Dataset reader '{target}' is not available; see "
            "dvlt.data.sources.datasets.scannetpp.ScanNetpp for a reference implementation."
        )

    def available_data_fields(self) -> List[DataField]:
        raise NotImplementedError

    def num_videos(self) -> int:
        raise NotImplementedError

    def num_views(self, video_idx: int) -> int:
        raise NotImplementedError

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        raise NotImplementedError

    def _read_data(self, video_idx, frame_idxs, view_idxs, data_fields):
        raise NotImplementedError
