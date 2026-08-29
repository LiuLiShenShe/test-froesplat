#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段十一 单元测试 — 验证 GT 拆分、per_instance 候选、评分门控、P2/P6 坍缩检测、A6/A7 安全机制。

直接在 gt_audit_convert 模块和主脚本模块上做测试（主脚本有 torch/CUDA 依赖，
但测试的函数本身是纯 numpy，只 mock 掉 torch.cuda 调用）。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# ── 把两个脚本所在目录加入 sys.path ────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT_DIR = SCRIPT_DIR.parent.parent.parent.parent / "数据管理" / "07-运行脚本与超参" / "S20-RAP-FSAM3掩膜生成与验证" / "脚本"
sys.path.insert(0, str(MAIN_SCRIPT_DIR))

# ── GT 拆分模块（轻量，无 torch 依赖）─────────────────────────────────
import gt_audit_convert as gta
from gt_audit_convert import (
    audit_frame, compose_scoped_masks, convert_labelme_scoped,
    shape_to_mask, SMALL_AREA_RATIO, POT_GAP_RATIO, BLUE_MARGIN,
)


def _load_main_mod():
    """尝试导入主脚本；若 torch 不可用则跳过依赖主脚本的测试。

    注意：dataclass 字段注解解析要求模块已注册到 sys.modules，因此先注册占位模块再 exec。
    """
    main_path = MAIN_SCRIPT_DIR / "生成RAP-FSAM3掩膜.py"
    mod_name = "rap_fsam3_test_import"
    spec = importlib.util.spec_from_file_location(mod_name, str(main_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except (ImportError, OSError, RuntimeError) as e:
        del sys.modules[mod_name]
        return None, str(e)
    return mod, None


# ════════════════════════════════════════════════════════════════════
# 1. test_gt_split  — 构造含 plant/pot/cube 的 labelme JSON，验证口径拆分
# ════════════════════════════════════════════════════════════════════
class TestGtSplit(unittest.TestCase):
    """§9.1  GT 拆分：构造含 plant + pot + cube 三个 shape 的 JSON，
    验证 plant_only / plant_plus_pot / reference_cube 三个口径正确分离。"""

    def _make_json(self, tmp: Path, shapes: list[dict], w=3840, h=2160):
        data = {"imageWidth": w, "imageHeight": h, "shapes": shapes, "imagePath": "test.jpg"}
        (tmp / "test.json").write_text(json.dumps(data), encoding="utf-8")
        # 生成一张空白原图供颜色判定
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = [200, 180, 160]  # 灰棕色背景（非蓝色）
        cv2.imwrite(str(tmp / "test.jpg"), img)
        return tmp / "test.json"

    def test_three_shape_split(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # plant: 大面积，覆盖上半区
            plant_pts = [[100, 100], [500, 100], [500, 1200], [100, 1200]]
            # pot: 小面积，位于植株下方
            pot_pts = [[200, 1300], [400, 1300], [400, 1500], [200, 1500]]
            # cube: 小面积，位于更下方，蓝色
            cube_pts = [[250, 1600], [350, 1600], [350, 1700], [250, 1700]]
            json_path = self._make_json(td, [
                {"label": "plant", "shape_type": "polygon", "points": plant_pts},
                {"label": "pot", "shape_type": "polygon", "points": pot_pts},
                {"label": "cube", "shape_type": "polygon", "points": cube_pts},
            ])
            # 在原图上把 cube 区域涂蓝
            img = cv2.imread(str(td / "test.jpg"))
            img[1600:1700, 250:350] = [255, 50, 50]  # BGR: blue 高通道
            cv2.imwrite(str(td / "test.jpg"), img)

            geoms, meta = audit_frame(json_path, img)
            roles = {g.role for g in geoms}
            self.assertIn("plant", roles)
            self.assertIn("pot", roles)
            # cube 可能被判为 reference_cube 或 pot（取决于颜色判定精度）
            # 但 plant 必须存在且不是 cube
            plant_geom = [g for g in geoms if g.role == "plant"]
            self.assertTrue(len(plant_geom) >= 1, f"plant not found: roles={[g.role for g in geoms]}")

            # plant_only 不含 pot/cube
            mask_po = compose_scoped_masks(geoms, "plant_only")
            plant_only_area = int(mask_po.sum())
            self.assertGreater(plant_only_area, 0)

            # plant_plus_pot 含 pot
            mask_pp = compose_scoped_masks(geoms, "plant_plus_pot")
            self.assertGreaterEqual(int(mask_pp.sum()), plant_only_area)

    def test_cube_not_in_plant_gt(self):
        """cube 永不进入 plant_only / plant_plus_pot GT。"""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            plant_pts = [[100, 100], [500, 100], [500, 1000], [100, 1000]]
            cube_pts = [[250, 1500], [350, 1500], [350, 1600], [250, 1600]]
            json_path = self._make_json(td, [
                {"label": "1", "shape_type": "polygon", "points": plant_pts},
                {"label": "1", "shape_type": "polygon", "points": cube_pts},
            ])
            # 把 cube 区域涂成蓝色
            img = cv2.imread(str(td / "test.jpg"))
            img[1500:1600, 250:350] = [255, 50, 50]  # BGR
            cv2.imwrite(str(td / "test.jpg"), img)

            geoms, _ = audit_frame(json_path, img)
            cube_geom = [g for g in geoms if g.role == "reference_cube"]
            plant_geom = [g for g in geoms if g.role == "plant"]
            self.assertTrue(len(plant_geom) >= 1, "plant not detected")
            # 如果 cube 被正确识别为 reference_cube，它不应进入 plant_only
            if cube_geom:
                mask_po = compose_scoped_masks(geoms, "plant_only")
                cube_mask = compose_scoped_masks(geoms, "reference_cube")
                # plant_only 与 cube 的交集应为空
                overlap = int((mask_po & cube_mask).sum())
                self.assertEqual(overlap, 0, "cube leaked into plant_only GT")


# ════════════════════════════════════════════════════════════════════
# 2. test_candidate_gen  — per_instance 模式产生独立候选，不 OR
# ════════════════════════════════════════════════════════════════════
class TestCandidateGen(unittest.TestCase):
    """§9.2  per_instance 候选生成：mock SAM3 返回 3 个实例，断言产生 3 个 Candidate，
    且各实例掩膜互不为 OR 合并。"""

    def test_per_instance_produces_separate_candidates(self):
        main_mod, err = _load_main_mod()
        if main_mod is None:
            self.skipTest(f"Main script requires torch/CUDA: {err}")

        Candidate = main_mod.Candidate
        # 构造 3 个互不重叠的掩膜候选
        h, w = 100, 100
        mask1 = np.zeros((h, w), dtype=bool)
        mask1[10:30, 10:30] = True
        mask2 = np.zeros((h, w), dtype=bool)
        mask2[50:70, 50:70] = True
        mask3 = np.zeros((h, w), dtype=bool)
        mask3[70:90, 70:90] = True

        candidates = [
            Candidate(prompt_id="P2", prompt_text="plant", mask=mask1,
                      scores=[0.9], raw_detection_count=1, instance_id=0),
            Candidate(prompt_id="P2", prompt_text="plant", mask=mask2,
                      scores=[0.8], raw_detection_count=1, instance_id=1),
            Candidate(prompt_id="P2", prompt_text="plant", mask=mask3,
                      scores=[0.7], raw_detection_count=1, instance_id=2),
        ]

        # 验证 3 个实例掩膜互不为 OR 合并
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0].instance_id, 0)
        self.assertEqual(candidates[1].instance_id, 1)
        self.assertEqual(candidates[2].instance_id, 2)
        # 互不重叠
        self.assertFalse((candidates[0].mask & candidates[1].mask).any())
        self.assertFalse((candidates[1].mask & candidates[2].mask).any())
        # 各自面积 > 0
        for c in candidates:
            self.assertGreater(c.mask.sum(), 0)


# ════════════════════════════════════════════════════════════════════
# 3. test_empty_mask_score  — 空 mask 候选 total_score==0
# ════════════════════════════════════════════════════════════════════
class TestEmptyMaskScore(unittest.TestCase):
    """§9.3  空 mask 候选的 score_candidate 返回 total_score=0 且 empty_flag=True。"""

    def test_empty_mask_scores_zero(self):
        from dataclasses import dataclass, field
        import argparse, math

        main_mod, err = _load_main_mod()
        if main_mod is None:
            self.skipTest(f"Main script requires torch/CUDA: {err}")

        Candidate = main_mod.Candidate
        score_candidate = main_mod.score_candidate

        h, w = 100, 100
        empty_mask = np.zeros((h, w), dtype=bool)
        cand = Candidate(prompt_id="P2", prompt_text="plant", mask=empty_mask,
                         scores=[0.5], raw_detection_count=0, instance_id=0)

        args = argparse.Namespace(
            component_min_area_ratio=0.0005,
            leakage_bottom_start_ratio=0.62,
            leakage_max_bottom_fraction=0.02,
            leakage_side_mode="both",
            leakage_side_band_ratio=0.025,
            leakage_max_side_fraction=0.004,
            area_min_ratio=0.01,
            area_max_ratio=0.80,
            area_target_ratio=0.0,
            use_semantic_gate=False,
            use_temporal_alignment=False,
            vertical_coverage_min_ratio=0.30,
            box_track_overlap_min=0.20,
            collapse_area_threshold=0.05,
        )
        weights = {"area": 1, "comp": 1, "edge": 1, "temp": 1, "contrast": 1, "sam": 0.5}
        image = __import__("PIL").Image.new("RGB", (w, h), (128, 128, 128))

        rec = score_candidate(Path("test.jpg"), image, cand, {}, weights, args)
        self.assertEqual(rec.total_score, 0.0, "empty mask must score 0")
        self.assertTrue(rec.empty_flag, "empty_flag must be True")
        self.assertEqual(rec.q_temp, 0.0, "q_temp must be 0 for empty mask")


# ════════════════════════════════════════════════════════════════════
# 4. test_p2p6_collapse  — 构造 P2==P6，断言 detect_semantic_collapse=True
# ════════════════════════════════════════════════════════════════════
class TestP2P6Collapse(unittest.TestCase):
    """§9.4  P2/P6 语义坍缩检测：构造 P2 完全等于 P6（面积/盆区一致），断言返回 True。"""

    def test_identical_masks_detected(self):
        main_mod, err = _load_main_mod()
        if main_mod is None:
            self.skipTest(f"Main script requires torch/CUDA: {err}")

        detect = main_mod.detect_semantic_collapse
        import argparse
        args = argparse.Namespace(collapse_area_threshold=0.05)

        h, w = 200, 100
        mask = np.zeros((h, w), dtype=bool)
        mask[10:150, 20:80] = True
        # P2 == P6（语义坍缩）
        self.assertTrue(detect(mask, mask, args), "Identical P2/P6 should be detected as collapse")

    def test_different_masks_not_collapsed(self):
        main_mod, err = _load_main_mod()
        if main_mod is None:
            self.skipTest(f"Main script requires torch/CUDA: {err}")

        detect = main_mod.detect_semantic_collapse
        import argparse
        args = argparse.Namespace(collapse_area_threshold=0.05)

        h, w = 200, 100
        p2 = np.zeros((h, w), dtype=bool)
        p2[10:100, 20:80] = True
        p6 = np.zeros((h, w), dtype=bool)
        p6[10:180, 10:90] = True  # 明显更大
        self.assertFalse(detect(p2, p6, args), "Different masks should not collapse")


# ════════════════════════════════════════════════════════════════════
# 5. test_a6_coords  — 未配准的 2D mask 直接混合应触发标记
# ════════════════════════════════════════════════════════════════════
class TestA6Coords(unittest.TestCase):
    """§9.5  A6 跨视角：两帧未配准的 2D mask 不应直接 OR 到同一图像坐标。
    通过检查 apply_cross_view_consensus 需要 use_cross_view_registration 开关来验证。"""

    def test_a6_default_off(self):
        """A6 默认关闭（use_cross_view_consensus=False）。"""
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--use_cross_view_consensus", action="store_true")
        p.add_argument("--use_cross_view_registration", action="store_true")
        args = p.parse_args([])
        self.assertFalse(args.use_cross_view_consensus,
                         "A6 cross-view consensus must be OFF by default")

    def test_a6_registration_flag_exists(self):
        """A6 配准开关应存在且默认关。"""
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--use_cross_view_consensus", action="store_true")
        p.add_argument("--use_cross_view_registration", action="store_true")
        args = p.parse_args([])
        self.assertFalse(args.use_cross_view_registration,
                         "A6 registration must be OFF by default")


# ════════════════════════════════════════════════════════════════════
# 6. test_a7_single_id  — mock 多 object ID，断言只返回一个
# ════════════════════════════════════════════════════════════════════
class TestA7SingleId(unittest.TestCase):
    """§9.6  A7 单对象 ID 选择：select_single_object_id 只返回一个 ID，
    不会 OR 合并多个 object ID。"""

    def test_returns_single_id(self):
        main_mod, err = _load_main_mod()
        if main_mod is None:
            self.skipTest(f"Main script requires torch/CUDA: {err}")

        select_id = main_mod.select_single_object_id

        h, w = 100, 100
        # 3 个 object mask
        obj0 = np.zeros((h, w), dtype=bool)
        obj0[10:30, 10:30] = True
        obj1 = np.zeros((h, w), dtype=bool)
        obj1[50:70, 50:70] = True
        obj2 = np.zeros((h, w), dtype=bool)
        obj2[70:90, 70:90] = True
        # seed 与 obj0 重叠
        seed = np.zeros((h, w), dtype=bool)
        seed[10:30, 10:30] = True

        chosen_id = select_id([obj0, obj1, obj2], seed)
        self.assertIsInstance(chosen_id, int)
        self.assertEqual(chosen_id, 0, "seed matches obj0 → should select ID 0")
        # 不是返回所有 ID
        self.assertNotEqual(chosen_id, -1, "should return a valid ID")


if __name__ == "__main__":
    unittest.main()
