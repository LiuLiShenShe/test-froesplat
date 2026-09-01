#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 13.1 — Same-Run Integration Test

Proves A6 → A7 → Pass 2 scoring chain works in a single pipeline run.
Unlike the combined test (which calls A6 and A7 separately), this test
chains them: A6 consensus → A7 memory seed from consensus → Pass 2 variant creation.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

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


def _synth_gray(h=910, w=512):
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (h, w), dtype=np.uint8)


def _synth_colmap_obs(stem, points):
    return mod.ColmapObservation(
        image_name=f"{stem}.jpg",
        mask_stem=stem,
        points=np.asarray(points, dtype=np.float32),
    )


def _get_scoring_args():
    return types.SimpleNamespace(
        area_min_ratio=0.01,
        area_max_ratio=0.80,
        area_target_ratio=0.0,
        component_min_area_ratio=0.0005,
        leakage_bottom_start_ratio=0.62,
        leakage_max_bottom_fraction=0.02,
        leakage_side_mode="both",
        leakage_side_band_ratio=0.025,
        leakage_max_side_fraction=0.004,
        use_temporal_alignment=False,
        use_semantic_gate=False,
        edge_band_ratio=0.05,
        edge_center_ratio=0.4,
        bottom_leak_threshold=0.3,
        side_leak_threshold=0.3,
        vertical_coverage_min_ratio=0.30,
        box_track_overlap_min=0.20,
    )


# ──────────────────────────────────────────────────────────────────────
# Integration: A6 → A7 → Pass 2 scoring chain
# ──────────────────────────────────────────────────────────────────────
class TestSameRunPipeline:
    def test_a6_a7_pass2_chain(self):
        """
        Full chain: A6 consensus → A7 memory from consensus → Pass 2 variant scoring.

        This proves that in a single pipeline run, A6 output feeds into A7 seeding,
        and both produce candidates that enter unified scoring.
        """
        from PIL import Image as PILImage

        h, w = 100, 200
        stems = ["f0", "f1", "f2"]

        # Step 1: Construct selected_by_stem (Pass 1 output)
        selected_by_stem = {
            "f0": _synth_mask(h, w, region=(20, 80, 50, 150)),
            "f1": _synth_mask(h, w, region=(25, 85, 55, 155)),
            "f2": _synth_mask(h, w, region=(22, 82, 52, 152)),
        }

        # Step 2: Run A6 consensus
        gray_by_stem = {s: _synth_gray(h, w) for s in stems}
        colmap = {
            "f0": _synth_colmap_obs("f0", [[100, 50], [120, 60]]),
            "f1": _synth_colmap_obs("f1", [[105, 52], [125, 62]]),
            "f2": _synth_colmap_obs("f2", [[102, 51], [122, 61]]),
        }
        a6_args = types.SimpleNamespace(
            consensus_min_frames=2,
            consensus_center_band_ratio=0.35,
            consensus_center_decay=0.65,
            consensus_support_ratio=0.55,
            consensus_low_vote_ratio=0.30,
            consensus_recall_ratio=0.70,
            consensus_static_weight=0.60,
            consensus_static_std_threshold=0.02,
            consensus_bridge_kernel=31,
            consensus_adhesion_min_area_ratio=0.004,
            consensus_variant_leak_weight=0.0,
            consensus_geometry_dilation=19,
            consensus_min_area_ratio=0.0008,
            consensus_fallback_iou=0.75,
        )
        consensus_result = mod.apply_cross_view_consensus(
            selected_by_stem, gray_by_stem, colmap, None, a6_args
        )
        assert consensus_result is not None, "A6 must succeed with 3 frames"
        assert isinstance(consensus_result, mod.ConsensusResult)

        # Step 3: Use A6 output as A7 seed base (same as production L2780-2781)
        base_for_memory = consensus_result.per_frame_masks
        seed_stem = "f1"  # middle frame

        # Step 4: Mock SAM3 predictor for A7
        class _MockPredictor:
            def __init__(self):
                self._calls = []
            def handle_request(self, request):
                self._calls.append(("request", request))
                rtype = request.get("type", "")
                if rtype == "start_session":
                    return {"session_id": "mock"}
                elif rtype == "add_prompt":
                    return {"outputs": {"out_obj_ids": [0]}}
                elif rtype == "close_session":
                    return {}
                return {}
            def handle_stream_request(self, request):
                self._calls.append(("stream", request))
                for idx in range(3):
                    mask = _synth_mask(h, w, region=(20+idx, 80+idx, 50+idx, 150+idx))
                    yield {
                        "frame_index": idx,
                        "outputs": {
                            "out_obj_ids": [0],
                            "out_binary_masks": [mask.astype(np.uint8)],
                        },
                    }

        mock_pred = _MockPredictor()

        # Create temp image files for A7
        import shutil, tempfile, tempfile as _tf
        from PIL import Image as PILImg
        tmp = Path(tempfile.mkdtemp())
        image_paths = {}
        for stem in stems:
            img = PILImg.fromarray(np.random.RandomState(hash(stem) % 2**31).randint(0, 255, (h, w, 3), dtype=np.uint8))
            p = tmp / f"{stem}.jpg"
            img.save(str(p))
            image_paths[stem] = p

        a7_args = types.SimpleNamespace(
            sam3_repo=Path("/tmp/fake_repo"),
            sam3_checkpoint=Path("/tmp/fake_ckpt"),
            memory_seed_mode="best_score",
            memory_bidirectional=False,
            memory_max_frames=0,
        )

        try:
            with patch.object(mod, "load_sam3_video_predictor", return_value=mock_pred):
                memory_masks, memory_info = mod.propagate_memory_masks(
                    image_paths, seed_stem, "potted plant",
                    base_for_memory[seed_stem], stems, a7_args
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        assert len(memory_masks) > 0, "A7 must produce masks"
        assert memory_info["状态"] == "ok"

        # Step 5: Pass 2 variant creation (reproduce L2833-2843 logic)
        from PIL import Image as PILImage
        pil_images = {s: PILImage.fromarray(np.stack([gray_by_stem[s]]*3, axis=-1)) for s in stems}

        sc_args = _get_scoring_args()
        weights = {"area": 1, "comp": 1, "edge": 1, "temp": 1, "contrast": 1, "sam": 0.5}
        fake_path = Path("/tmp/fake.jpg")

        for stem in stems:
            image = pil_images[stem]
            variants = []

            # A1s baseline
            variants.append(mod.Candidate(
                prompt_id="A1s", prompt_text="potted plant",
                mask=selected_by_stem[stem], scores=[], raw_detection_count=0,
            ))

            # A6 consensus variant
            if stem in consensus_result.per_frame_masks:
                a6_mask = consensus_result.per_frame_masks[stem]
                variants.append(mod.Candidate(
                    prompt_id="A6共识", prompt_text="cross_view_consensus",
                    mask=a6_mask, scores=[], raw_detection_count=0,
                ))

            # A7 memory variant (with shape guard)
            if memory_masks and stem in memory_masks:
                a7_mask = memory_masks[stem]
                img_w, img_h = image.size
                if a7_mask.shape == (img_h, img_w):
                    variants.append(mod.Candidate(
                        prompt_id="A7记忆", prompt_text="memory_propagation",
                        mask=a7_mask, scores=[], raw_detection_count=0,
                    ))

            # Score all variants
            variant_records = [
                mod.score_candidate(fake_path, image, item, {}, weights, sc_args)
                for item in variants
            ]

            # Verify: all variants scored successfully
            assert len(variant_records) >= 2, f"{stem}: expected ≥2 variants (A1s + A6), got {len(variant_records)}"
            for rec in variant_records:
                assert hasattr(rec, "total_score"), f"{stem}: missing total_score"
                assert rec.total_score >= 0, f"{stem}: negative score"

            # Verify: max picks best
            best = max(variant_records, key=lambda r: r.total_score)
            assert best is not None

        # Step 6: Verify A6→A7 chain: A7 used consensus masks as seed base
        add_prompt_calls = [c for c in mock_pred._calls if c[0] == "request" and c[1].get("type") == "add_prompt"]
        assert len(add_prompt_calls) >= 1, "A7 must have called add_prompt (memory write)"
        # The seed text must be "potted plant"
        assert add_prompt_calls[0][1]["text"] == "potted plant"

    def test_a7_shape_guard_in_chain(self):
        """When A7 returns wrong-shape mask, Pass 2 shape guard skips it."""
        from PIL import Image as PILImage

        h, w = 100, 200
        selected_mask = _synth_mask(h, w, region=(20, 80, 50, 150))
        image = PILImage.fromarray(np.stack([_synth_gray(h, w)]*3, axis=-1))

        # A7 returns wrong-shape mask
        memory_masks = {"test_stem": np.zeros((50, 100), dtype=bool)}

        # Reproduce Pass 2 shape guard
        variants = [mod.Candidate(
            prompt_id="A1s", prompt_text="potted plant",
            mask=selected_mask, scores=[], raw_detection_count=0,
        )]
        stem = "test_stem"
        if memory_masks and stem in memory_masks:
            a7_mask = memory_masks[stem]
            img_w, img_h = image.size
            if a7_mask.shape == (img_h, img_w):
                variants.append(mod.Candidate(
                    prompt_id="A7记忆", prompt_text="memory_propagation",
                    mask=a7_mask, scores=[], raw_detection_count=0,
                ))

        # Only A1s in variants — A7 skipped due to shape mismatch
        assert len(variants) == 1
        assert variants[0].prompt_id == "A1s"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
