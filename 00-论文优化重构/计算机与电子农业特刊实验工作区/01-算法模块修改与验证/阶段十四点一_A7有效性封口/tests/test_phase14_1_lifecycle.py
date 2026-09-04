#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 14.1 — A7 lifecycle and diagnostic tests

Covers two fixes applied to the production pipeline:
1. Strict session lifecycle: close_session is called on EVERY exit path
   (success, seed_empty, OOM, exception) — previously OOM/exception leaked
   sessions and accumulated GPU memory.
2. Per-sample A7 diagnostics: _merge_sampled_memory_info() aggregates
   per-sample status into a summary grounded in ALL samples (previously only
   the LAST sample's info survived), and propagated/scored/selected counts
   are kept distinct.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

PIPELINE = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
                "/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py")


def _load():
    name = "p141_gen"
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


class _MockPredictor:
    """Simulates SAM3 video predictor for lifecycle tests."""

    def __init__(self, mode="ok", raise_on="none"):
        self._mode = mode
        self._raise_on = raise_on
        self._calls = []

    def handle_request(self, request):
        self._calls.append(("request", request))
        rtype = request.get("type", "")
        if rtype == "start_session":
            if self._raise_on == "start_session":
                if self._mode == "oom":
                    raise torch.cuda.OutOfMemoryError("OOM in start_session")
                raise RuntimeError("Exception in start_session")
            return {"session_id": "mock_session"}
        if rtype == "add_prompt":
            if self._raise_on == "add_prompt":
                if self._mode == "oom":
                    raise torch.cuda.OutOfMemoryError("OOM in add_prompt")
                raise RuntimeError("Exception in add_prompt")
            if self._mode == "seed_empty":
                return {"outputs": {"out_obj_ids": []}}
            return {"outputs": {"out_obj_ids": [0]}}
        if rtype == "close_session":
            return {}
        return {}

    def handle_stream_request(self, request):
        self._calls.append(("stream", request))
        if self._raise_on == "propagate":
            if self._mode == "oom":
                raise torch.cuda.OutOfMemoryError("OOM in propagate")
            raise RuntimeError("Exception in propagate")
        for idx in range(3):
            mask = _synth_mask(100, 200, region=(20 + idx, 80 + idx, 50 + idx, 150 + idx))
            yield {
                "frame_index": idx,
                "outputs": {
                    "out_obj_ids": [0],
                    "out_binary_masks": [mask.astype(np.uint8)],
                },
            }


class TestPropagateMemoryMasksLifecycle:
    """Verify session lifecycle: close_session called on every exit path."""

    def _run(self, mode="ok", raise_on="none"):
        h, w = 100, 200
        stems = ["f0", "f1", "f2"]
        seed_mask = _synth_mask(h, w, region=(20, 80, 50, 150))
        tmp = Path(tempfile.mkdtemp())
        image_paths = {}
        try:
            for s in stems:
                img = np.random.RandomState(42).randint(0, 255, (h, w, 3), dtype=np.uint8)
                from PIL import Image as PILImg
                p = tmp / f"{s}.jpg"
                PILImg.fromarray(img).save(str(p))
                image_paths[s] = p

            pred = _MockPredictor(mode=mode, raise_on=raise_on)
            args = types.SimpleNamespace(
                memory_bidirectional=False,
                memory_max_frames=0,
                memory_seed_mode="best_score",
                sam3_repo=Path("/tmp/fake_repo"),
                sam3_checkpoint=Path("/tmp/fake_ckpt"),
            )
            masks, info = mod.propagate_memory_masks(
                image_paths, "f1", "potted plant", seed_mask, stems, args, predictor=pred
            )
            return masks, info, pred
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _close_count(self, pred):
        return [c for c in pred._calls if c[0] == "request" and c[1].get("type") == "close_session"]

    def test_close_on_success(self):
        masks, info, pred = self._run("ok")
        assert info["状态"] == "ok"
        assert len(self._close_count(pred)) == 1
        assert len(masks) == 3
        assert "memory" in info

    def test_close_on_seed_empty(self):
        masks, info, pred = self._run("seed_empty")
        assert info["状态"] == "seed_empty"
        assert masks == {}
        assert len(self._close_count(pred)) == 1

    def test_close_on_oom(self):
        masks, info, pred = self._run("oom", "propagate")
        assert info["状态"] == "cuda_oom_fallback"
        assert masks == {}
        assert len(self._close_count(pred)) == 1

    def test_close_on_oom_during_add_prompt(self):
        masks, info, pred = self._run("oom", "add_prompt")
        assert info["状态"] == "cuda_oom_fallback"
        assert masks == {}
        assert len(self._close_count(pred)) == 1

    def test_close_on_exception(self):
        masks, info, pred = self._run("exception", "propagate")
        assert info["状态"].startswith("unavailable:")
        assert masks == {}
        assert len(self._close_count(pred)) == 1

    def test_memory_dict_populated(self):
        masks, info, pred = self._run("ok")
        mem = info["memory"]
        for key in ("before_predictor_load", "after_start_session", "after_close_session"):
            assert key in mem
        assert isinstance(mem["after_start_session"]["allocated_mib"], float)

    def test_seed_stats_populated(self):
        masks, info, pred = self._run("ok")
        assert "种子面积比例" in info
        assert "种子框比例" in info
        assert info["种子面积比例"] > 0
        assert info["种子物体数"] == 1


class TestMergeSampledMemoryInfo:
    """Test the per-sample A7 aggregate helper directly (no main() needed)."""

    def _diag(self, status, n_masks=2, seed="S1_0000"):
        return {
            "seed_stem": seed,
            "帧数": 4,
            "状态": status,
            "传播掩膜数": n_masks,
            "种子面积比例": 0.05,
            "种子框比例": 0.02,
            "种子物体数": 1,
            "memory": {},
            "传播帧": [f"{seed}_f{i}" for i in range(n_masks)],
        }

    def test_all_ok_single_sample(self):
        samples = {"S1": self._diag("ok", n_masks=3)}
        out = mod._merge_sampled_memory_info(samples, {"S1": ["S1_0000", "S1_0025"]})
        assert out["状态"] == "ok"
        assert out["样本数"] == 1
        assert out["samples_ok"] == 1
        assert out["samples_oom"] == 0
        assert out["samples_unavailable"] == 0
        assert out["propagated_total"] == 3
        assert out["candidate_total"] == 3  # every valid mask enters Pass-2 scoring
        assert out["selected_total"] == 0   # filled by Pass-2; distinct from propagated
        # Single sample -> report its real seed stem, not "multi_sample"
        assert out["种子帧"] == "S1_0000"

    def test_all_ok_multi_sample_seed_flag(self):
        samples = {
            "S1": self._diag("ok", n_masks=2),
            "S2": self._diag("ok", n_masks=4),
        }
        out = mod._merge_sampled_memory_info(samples, {"S1": [], "S2": []})
        assert out["状态"] == "ok"
        assert out["样本数"] == 2
        assert out["samples_ok"] == 2
        assert out["propagated_total"] == 6
        assert out["candidate_total"] == 6
        # Multiple samples -> no single 种子帧 can represent all of them
        assert out["种子帧"] == "multi_sample"

    def test_one_oom_yields_partial_oom(self):
        samples = {
            "S1": self._diag("cuda_oom_fallback", n_masks=0),
            "S2": self._diag("ok", n_masks=2),
        }
        out = mod._merge_sampled_memory_info(samples, {"S1": [], "S2": []})
        assert out["状态"] == "partial_oom"
        assert out["samples_ok"] == 1
        assert out["samples_oom"] == 1
        assert out["propagated_total"] == 2  # only the ok sample's masks count

    def test_one_unavailable_yields_partial_unavailable(self):
        samples = {
            "S1": self._diag("ok", n_masks=1),
            "S2": self._diag("unavailable: RuntimeError: boom", n_masks=0),
        }
        out = mod._merge_sampled_memory_info(samples, {"S1": [], "S2": []})
        assert out["状态"] == "partial_unavailable"
        assert out["samples_unavailable"] == 1

    def test_one_seed_empty_yields_partial_seed_empty(self):
        samples = {
            "S1": self._diag("seed_empty", n_masks=0),
            "S2": self._diag("ok", n_masks=3),
        }
        out = mod._merge_sampled_memory_info(samples, {"S1": [], "S2": []})
        assert out["状态"] == "partial_seed_empty"
        assert out["samples_seed_empty"] == 1

    def test_oom_priority_over_seed_empty(self):
        # oom is the stronger failure signal and must win the summary status
        samples = {
            "S1": self._diag("seed_empty", n_masks=0),
            "S2": self._diag("cuda_oom_fallback", n_masks=0),
        }
        out = mod._merge_sampled_memory_info(samples, {"S1": [], "S2": []})
        assert out["状态"] == "partial_oom"
        assert out["samples_seed_empty"] == 1
        assert out["samples_oom"] == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
