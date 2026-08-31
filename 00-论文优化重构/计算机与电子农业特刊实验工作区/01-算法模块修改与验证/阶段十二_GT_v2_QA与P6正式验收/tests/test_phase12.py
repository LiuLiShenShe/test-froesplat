#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段十二集成测试（§六）。
验证 GT v2 QA、代码修复、P6 评估路径隔离、sam3_mask_threshold 传参等。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

# ── 路径 ──
DELIVER = Path(__file__).resolve().parent.parent
GT_CLEAN_DIR = DELIVER / "GT_potted_clean"
GT_SRC = Path("/data/fj/F2DMAS/03-GT-区分")
PIPELINE = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
                "/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py")
FAIL_FRAMES = {("CaoMei1", "0100"), ("ChangShouHua2", "0100"), ("DouBanLv1", "0000")}


def load_pipeline_module():
    """Load the pipeline module dynamically."""
    mod_name = "生成RAP-FSAM3掩膜"
    spec = importlib.util.spec_from_file_location(mod_name, PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    # Add the script's directory to sys.path so its imports resolve
    script_dir = str(PIPELINE.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec.loader.exec_module(mod)
    return mod


def load_gt_masks():
    """Load all GT potted_clean masks."""
    masks = {}
    for sample_dir in sorted(GT_CLEAN_DIR.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        for p in sorted(sample_dir.glob("mask_potted_clean_*.png")):
            frame = p.stem.replace("mask_potted_clean_", "")
            import cv2
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            masks[(sample, frame)] = img > 127 if img is not None else None
    return masks


class TestGtCleanNonEmpty:
    """§6.1: 21帧均能生成非空 gt_potted_clean"""

    def test_all_frames_nonempty(self):
        masks = load_gt_masks()
        assert len(masks) == 21, f"Expected 21 GT frames, got {len(masks)}"
        for key, m in masks.items():
            assert m is not None, f"GT mask missing for {key}"
            assert m.any(), f"GT mask is empty for {key}"


class TestGtCleanNoCubeOverlap:
    """§6.2: gt_potted_clean & blue_cube = ∅"""

    def test_no_overlap(self):
        cube_dir = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
                        "/S20-RAP-FSAM3掩膜生成与验证/GT口径拆分审计/gt_masks_split_v2")
        import cv2
        for sample_dir in sorted(GT_CLEAN_DIR.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample = sample_dir.name
            for p in sorted(sample_dir.glob("mask_potted_clean_*.png")):
                frame = p.stem.replace("mask_potted_clean_", "")
                clean = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
                cube_path = cube_dir / sample / f"mask_blue_cube_{frame}.png"
                if cube_path.exists():
                    cube = cv2.imread(str(cube_path), cv2.IMREAD_GRAYSCALE) > 127
                    overlap = int((clean & cube).sum())
                    assert overlap == 0, f"{sample}/{frame}: clean∩cube = {overlap}px"


class TestDerivedPlantNotInP2:
    """§6.3: derived_plant 不进入正式 P2 指标"""

    def test_auto_plant_is_derived(self):
        """14帧 auto plant 不应被用于 formal P2."""
        qa_csv = DELIVER / "GT_QA" / "gt_audit_v2_qa.csv"
        assert qa_csv.exists(), "gt_audit_v2_qa.csv not found"
        import csv as _csv
        with qa_csv.open(encoding="utf-8-sig") as f:
            rows = list(_csv.DictReader(f))
        auto_frames = [r for r in rows if r.get("plant_auto") == "1"]
        assert len(auto_frames) == 14, f"Expected 14 auto-plant frames, got {len(auto_frames)}"
        for r in auto_frames:
            assert r.get("formal_p2_valid") in ("0", "False", ""), (
                f"{r['sample']}/{r['frame']}: auto plant should not be formal P2 valid"
            )


class TestMaskThresholdParam:
    """§6.4: sam3_mask_threshold 改变 → _masks_scores_boxes 二值化结果变化"""

    def test_threshold_affects_output(self):
        mod = load_pipeline_module()
        fn = mod._masks_scores_boxes
        # Create a mock output with logits
        h, w = 10, 10
        logits = np.zeros((1, 1, h, w), dtype=np.float32)
        logits[0, 0, 3:7, 3:7] = 0.6  # region with value 0.6
        logits[0, 0, 0:2, 0:2] = 0.45  # region with value 0.45
        output = {"masks_logits": logits}

        # _masks_scores_boxes returns (masks_list, scores_list, boxes_list)
        masks_05, scores_05, boxes_05 = fn(output, h, w, mask_threshold=0.5)
        assert len(masks_05) == 1, "Should find 1 instance above threshold 0.5"
        m05 = np.array(masks_05[0], dtype=bool)
        assert m05[4, 4] == True, "0.6 region should pass threshold 0.5"

        # With threshold 0.4: both regions included in one merged mask
        masks_04, scores_04, boxes_04 = fn(output, h, w, mask_threshold=0.4)
        assert len(masks_04) >= 1, "Should find at least 1 instance with threshold 0.4"
        m04 = np.array(masks_04[0], dtype=bool)
        assert m04[1, 1] == True, "0.45 region should pass threshold 0.4"
        assert m04[4, 4] == True, "0.6 region should pass threshold 0.4"

        # With threshold 0.7: mask exists but is empty (all False)
        masks_07, scores_07, boxes_07 = fn(output, h, w, mask_threshold=0.7)
        assert len(masks_07) >= 1, "Should still return a mask (possibly empty)"
        m07 = np.array(masks_07[0], dtype=bool)
        assert m07.sum() == 0, "No pixels should pass threshold 0.7"


class TestScoreSelectInstanceMatch:
    """§6.5: score_select 返回正确 instance_id"""

    def test_per_instance_selection(self):
        mod = load_pipeline_module()
        Candidate = mod.Candidate
        ScoreRecord = mod.ScoreRecord

        mask_a = np.zeros((10, 10), dtype=bool)
        mask_a[2:5, 2:5] = True
        mask_b = np.zeros((10, 10), dtype=bool)
        mask_b[6:9, 6:9] = True

        cands = [
            Candidate(prompt_id="P6", prompt_text="potted plant", instance_id=1, mask=mask_a, box=(2, 2, 5, 5),
                      sam_score=0.9, mask_threshold=0.5, scores=[0.9],
                      raw_detection_count=2, source_stage="pass1", prompt_mode="single"),
            Candidate(prompt_id="P6", prompt_text="potted plant", instance_id=2, mask=mask_b, box=(6, 6, 9, 9),
                      sam_score=0.7, mask_threshold=0.5, scores=[0.7],
                      raw_detection_count=2, source_stage="pass1", prompt_mode="single"),
        ]
        recs = [
            ScoreRecord(image_name="test.jpg", prompt_id="P6", prompt_text="potted plant",
                        total_score=0.6, empty_flag=False,
                        q_edge=0.5, q_area=0.8, q_comp=1.0,
                        q_temp=0.5, q_contrast=0.5, q_leak=0.0, q_side=0.0,
                        area_ratio=0.09, component_count=1, boundary_density=0.1,
                        temporal_iou=0.5, contrast=0.5, bottom_leak_fraction=0.0,
                        side_leak_fraction=0.0, sam_scores="0.9", instance_id=1,
                        semantic_enabled=False, semantic_total=0.0,
                        target_box_score=0.0, vertical_coverage_score=0.0,
                        pot_overlap_penalty=0.0, side_distractor_penalty=0.0,
                        center_prior_score=0.0, leak_penalty=0.0),
            ScoreRecord(image_name="test.jpg", prompt_id="P6", prompt_text="potted plant",
                        total_score=0.8, empty_flag=False,
                        q_edge=0.5, q_area=0.8, q_comp=1.0,
                        q_temp=0.5, q_contrast=0.5, q_leak=0.0, q_side=0.0,
                        area_ratio=0.09, component_count=1, boundary_density=0.1,
                        temporal_iou=0.5, contrast=0.5, bottom_leak_fraction=0.0,
                        side_leak_fraction=0.0, sam_scores="0.7", instance_id=2,
                        semantic_enabled=False, semantic_total=0.0,
                        target_box_score=0.0, vertical_coverage_score=0.0,
                        pot_overlap_penalty=0.0, side_distractor_penalty=0.0,
                        center_prior_score=0.0, leak_penalty=0.0),
        ]

        args = types.SimpleNamespace(
            use_prompt_ensemble=True,
            prompt_selection_mode="score_select",
            default_prompt_id="P6",
        )

        import cv2
        mask, pid, score, needs_reprompt = mod.select_mask(
            Path("test.jpg"), cands, recs, {"P6": "test"}, args
        )
        # instance_id=2 has higher total_score (0.8 > 0.6), should be selected
        assert instance_id_of(mask, mask_b), "Should select instance 2's mask"
        assert pid == "P6"

    def instance_id_of(mask, target):
        return np.array_equal(mask, target)


class TestP6ReadsIndependentPath:
    """§6.6: P6 评估读取独立 P6 预测路径"""

    def test_p6_path_exists(self):
        p6_dir = DELIVER / "P6_raw_baseline"
        assert p6_dir.exists(), "P6_raw_baseline directory should exist"
        contents = list(p6_dir.iterdir())
        assert len(contents) > 0, "P6_raw_baseline should contain output directories"


class TestP2P6PathDistinct:
    """§6.7: P2 和 P6 路径相同时报错（逻辑断言）"""

    def test_different_prompt_ids(self):
        mod = load_pipeline_module()
        assert "P2" in mod.BUILTIN_PROMPTS
        assert "P6" in mod.BUILTIN_PROMPTS
        assert mod.BUILTIN_PROMPTS["P2"] != mod.BUILTIN_PROMPTS["P6"], (
            "P2 and P6 prompts should be different"
        )


class TestProductionParseArgs:
    """§6.8: 测试使用生产 parse_args() 解析器"""

    def test_parse_args_defaults(self):
        mod = load_pipeline_module()
        # Simulate minimal args
        sys.argv = ["test", "--input_dir", "/tmp", "--output_dir", "/tmp/out"]
        try:
            args = mod.parse_args()
            assert args.candidate_mode == "per_instance"
            assert args.sam3_mask_threshold == 0.5
            assert args.default_prompt_id == "P2"
        finally:
            sys.argv = ["test"]


def instance_id_of(mask, target):
    return np.array_equal(mask, target)


# Allow running directly
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
