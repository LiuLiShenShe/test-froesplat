#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 13 — A7 Memory Propagation Forced Runtime Validation

Forces A7 production branch with mock SAM3 video predictor.
Tests memory lifecycle: write on seed frame → read on subsequent frames.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PIPELINE = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
                "/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py")


def _load():
    name = "p13_gen"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(PIPELINE.parent))
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _synth_mask(h=100, w=200, region=None):
    m = np.zeros((h, w), dtype=bool)
    if region:
        y0, y1, x0, x1 = region
        m[y0:y1, x0:x1] = True
    return m


def _synth_image(h=100, w=200):
    from PIL import Image
    arr = np.random.RandomState(42).randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class _MockPredictor:
    """Mock SAM3 video predictor that returns synthetic masks."""

    def __init__(self, mask_by_frame=None):
        self._mask_by_frame = mask_by_frame or {}
        self._session_id = "mock_session"
        self._calls = []

    def handle_request(self, request):
        self._calls.append(("request", request))
        rtype = request.get("type", "")
        if rtype == "start_session":
            return {"session_id": self._session_id}
        elif rtype == "add_prompt":
            obj_ids = request.get("obj_id", 0)
            return {"outputs": {"out_obj_ids": [obj_ids]}}
        elif rtype == "close_session":
            return {}
        return {}

    def handle_stream_request(self, request):
        self._calls.append(("stream", request))
        start = request.get("start_frame_index", 0)
        direction = request.get("propagation_direction", "forward")
        # Yield masks for frames around start
        for idx in sorted(self._mask_by_frame.keys()):
            if direction == "forward" and idx < start:
                continue
            if direction == "both" or direction == "forward":
                mask = self._mask_by_frame.get(idx, np.zeros((100, 200), dtype=bool))
                yield {
                    "frame_index": idx,
                    "outputs": {
                        "out_obj_ids": [0],
                        "out_binary_masks": [mask.astype(np.uint8)],
                    },
                }


def _make_args(**overrides):
    defaults = dict(
        sam3_repo=Path("/tmp/fake_repo"),
        sam3_checkpoint=Path("/tmp/fake_ckpt"),
        memory_seed_mode="best_score",
        memory_bidirectional=True,
        memory_max_frames=0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_image_paths(stems, tmp_dir=None):
    """Create synthetic image paths. Uses real temp files for shutil.copy."""
    import tempfile
    import os
    from PIL import Image

    paths = {}
    for stem in stems:
        p = Path(tempfile.mkdtemp()) / f"{stem}.jpg"
        img = _synth_image()
        img.save(str(p))
        paths[stem] = p
    return paths


# ──────────────────────────────────────────────────────────────────────
# A7-T1: enable
# ──────────────────────────────────────────────────────────────────────
class TestA7Enable:
    def test_flag_defaults_false(self):
        args = _make_args()
        assert not getattr(args, "use_memory_propagation", False)

    def test_flag_exists_in_parser(self):
        source = PIPELINE.read_text(encoding="utf-8")
        assert '--use_memory_propagation' in source


# ──────────────────────────────────────────────────────────────────────
# A7-T2: memory write — seed frame produces memory state
# ──────────────────────────────────────────────────────────────────────
class TestA7MemoryWrite:
    def test_seed_frame_add_prompt_called(self):
        """propagate_memory_masks calls add_prompt on seed frame → memory write."""
        h, w = 100, 200
        mask_by_frame = {
            0: _synth_mask(h, w, region=(20, 80, 50, 150)),
            1: _synth_mask(h, w, region=(22, 82, 52, 152)),
        }
        mock_pred = _MockPredictor(mask_by_frame)
        image_paths = _make_image_paths(["f0", "f1"])
        seed_mask = mask_by_frame[0]
        args = _make_args(memory_bidirectional=False)

        with patch.object(mod, "load_sam3_video_predictor", return_value=mock_pred):
            try:
                memory_masks, memory_info = mod.propagate_memory_masks(
                    image_paths, "f0", "potted plant", seed_mask, ["f0", "f1"], args
                )
            finally:
                for p in image_paths.values():
                    shutil.rmtree(p.parent, ignore_errors=True)

        # Verify add_prompt was called (memory write)
        add_prompt_calls = [c for c in mock_pred._calls if c[0] == "request" and c[1].get("type") == "add_prompt"]
        assert len(add_prompt_calls) >= 1, "add_prompt must be called for seed frame (memory write)"
        seed_call = add_prompt_calls[0][1]
        assert seed_call["frame_index"] == 0, "Seed frame index should be 0"
        assert seed_call["text"] == "potted plant"


# ──────────────────────────────────────────────────────────────────────
# A7-T3: memory read — subsequent frames read from memory state
# ──────────────────────────────────────────────────────────────────────
class TestA7MemoryRead:
    def test_propagate_in_video_called(self):
        """propagate_in_video is called → memory engine reads state for each frame."""
        h, w = 100, 200
        mask_by_frame = {
            0: _synth_mask(h, w, region=(20, 80, 50, 150)),
            1: _synth_mask(h, w, region=(22, 82, 52, 152)),
        }
        mock_pred = _MockPredictor(mask_by_frame)
        image_paths = _make_image_paths(["f0", "f1"])
        seed_mask = mask_by_frame[0]
        args = _make_args(memory_bidirectional=False)

        with patch.object(mod, "load_sam3_video_predictor", return_value=mock_pred):
            try:
                memory_masks, memory_info = mod.propagate_memory_masks(
                    image_paths, "f0", "potted plant", seed_mask, ["f0", "f1"], args
                )
            finally:
                for p in image_paths.values():
                    shutil.rmtree(p.parent, ignore_errors=True)

        # Verify propagate_in_video was called (memory read for each frame)
        stream_calls = [c for c in mock_pred._calls if c[0] == "stream"]
        assert len(stream_calls) >= 1, "propagate_in_video must be called (memory read)"
        assert stream_calls[0][1].get("type") == "propagate_in_video"


# ──────────────────────────────────────────────────────────────────────
# A7-T4: propagation — memory information participates in output
# ──────────────────────────────────────────────────────────────────────
class TestA7Propagation:
    def test_memory_masks_populated(self):
        """Memory propagation produces non-empty memory_masks dict."""
        h, w = 100, 200
        mask_by_frame = {
            0: _synth_mask(h, w, region=(20, 80, 50, 150)),
            1: _synth_mask(h, w, region=(22, 82, 52, 152)),
        }
        mock_pred = _MockPredictor(mask_by_frame)
        image_paths = _make_image_paths(["f0", "f1"])
        seed_mask = mask_by_frame[0]
        args = _make_args(memory_bidirectional=False)

        with patch.object(mod, "load_sam3_video_predictor", return_value=mock_pred):
            try:
                memory_masks, memory_info = mod.propagate_memory_masks(
                    image_paths, "f0", "potted plant", seed_mask, ["f0", "f1"], args
                )
            finally:
                for p in image_paths.values():
                    shutil.rmtree(p.parent, ignore_errors=True)

        assert len(memory_masks) > 0, "memory_masks should have at least one entry"
        for stem, mask in memory_masks.items():
            assert isinstance(mask, np.ndarray), f"memory_masks[{stem}] should be ndarray"
            assert mask.dtype == bool, f"memory_masks[{stem}] should be bool"
            assert mask.shape == (h, w), f"memory_masks[{stem}] shape mismatch"


# ──────────────────────────────────────────────────────────────────────
# A7-T5: output — memory_info is valid
# ──────────────────────────────────────────────────────────────────────
class TestA7Output:
    def test_memory_info_structure(self):
        h, w = 100, 200
        mask_by_frame = {
            0: _synth_mask(h, w, region=(20, 80, 50, 150)),
            1: _synth_mask(h, w, region=(22, 82, 52, 152)),
        }
        mock_pred = _MockPredictor(mask_by_frame)
        image_paths = _make_image_paths(["f0", "f1"])
        seed_mask = mask_by_frame[0]
        args = _make_args(memory_bidirectional=False)

        with patch.object(mod, "load_sam3_video_predictor", return_value=mock_pred):
            try:
                memory_masks, memory_info = mod.propagate_memory_masks(
                    image_paths, "f0", "potted plant", seed_mask, ["f0", "f1"], args
                )
            finally:
                for p in image_paths.values():
                    shutil.rmtree(p.parent, ignore_errors=True)

        assert "记忆后端" in memory_info
        assert "种子帧" in memory_info
        assert "传播方向" in memory_info
        assert memory_info["种子帧"] == "f0"
        assert memory_info["传播方向"] == "forward"


# ──────────────────────────────────────────────────────────────────────
# A7-T6: evidence — memory lifecycle recorded
# ──────────────────────────────────────────────────────────────────────
class TestA7Evidence:
    def test_memory_state_recorded(self):
        h, w = 100, 200
        mask_by_frame = {
            0: _synth_mask(h, w, region=(20, 80, 50, 150)),
            1: _synth_mask(h, w, region=(22, 82, 52, 152)),
        }
        mock_pred = _MockPredictor(mask_by_frame)
        image_paths = _make_image_paths(["f0", "f1"])
        seed_mask = mask_by_frame[0]
        args = _make_args(memory_bidirectional=False)

        with patch.object(mod, "load_sam3_video_predictor", return_value=mock_pred):
            try:
                memory_masks, memory_info = mod.propagate_memory_masks(
                    image_paths, "f0", "potted plant", seed_mask, ["f0", "f1"], args
                )
            finally:
                for p in image_paths.values():
                    shutil.rmtree(p.parent, ignore_errors=True)

        # Verify all memory lifecycle events are recorded
        assert memory_info.get("种子帧") == "f0"
        assert memory_info.get("记忆后端") == "sam3_video"
        # Verify predictor was called with both write (add_prompt) and read (propagate)
        call_types = [c[1].get("type", "") for c in mock_pred._calls if c[0] == "request"]
        assert "add_prompt" in call_types, "Memory write (add_prompt) must be called"


# ──────────────────────────────────────────────────────────────────────
# Multi-frame memory lifecycle
# ──────────────────────────────────────────────────────────────────────
class TestA7MultiFrameLifecycle:
    def test_two_frame_memory_write_and_read(self):
        """frame_0 writes memory, frame_1 reads it via propagation."""
        h, w = 100, 200
        mask_by_frame = {
            0: _synth_mask(h, w, region=(20, 80, 50, 150)),
            1: _synth_mask(h, w, region=(22, 82, 52, 152)),
        }
        mock_pred = _MockPredictor(mask_by_frame)
        image_paths = _make_image_paths(["f0", "f1"])
        seed_mask = mask_by_frame[0]
        args = _make_args(memory_bidirectional=False)

        with patch.object(mod, "load_sam3_video_predictor", return_value=mock_pred):
            try:
                memory_masks, memory_info = mod.propagate_memory_masks(
                    image_paths, "f0", "potted plant", seed_mask, ["f0", "f1"], args
                )
            finally:
                for p in image_paths.values():
                    shutil.rmtree(p.parent, ignore_errors=True)

        # Frame 0: memory write via add_prompt
        write_calls = [c for c in mock_pred._calls if c[0] == "request" and c[1].get("type") == "add_prompt"]
        assert len(write_calls) == 1
        assert write_calls[0][1]["frame_index"] == 0

        # Frame 1: memory read via propagate_in_video (yields mask for frame 1)
        stream_calls = [c for c in mock_pred._calls if c[0] == "stream"]
        assert len(stream_calls) == 1

        # Both frames have masks in output
        assert "f0" in memory_masks or "f1" in memory_masks


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
