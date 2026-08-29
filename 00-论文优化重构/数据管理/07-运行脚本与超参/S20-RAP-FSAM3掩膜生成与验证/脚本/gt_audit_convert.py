#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT 审计与口径拆分脚本（ForeSplat 修复 · 阶段十一 §二）

问题：实验一脚本 convert_labelme 把 LabelMe GT 中所有 shape 无条件 OR 成一张掩膜，
无法区分 植株 / 花盆 / 蓝色标定块。这导致：
  - DouBanLv1_0000 的 GT 同时含"植株 + 花盆"，但 P2（去盆）方法被错误判为失败；
  - 评测时无法分别报告 pot_fp / cube_fp / dark_leaf_recall。

本脚本用**纯几何 + 原图颜色**启发式（无样本名硬编码）把每个 shape 推断为：
  - plant         植株（通常面积最大、向上延展）
  - pot           花盆/托盘（小面积、位于植株主体下方、暖色）
  - reference_cube  蓝色标定块（小面积、位于下方、蓝色）
  - ignore         未识别（计入 pending 待人工复核）

产出：
  - gt_masks_split/<sample>/mask_plant_only_<frame>.png
  - gt_masks_split/<sample>/mask_plant_plus_pot_<frame>.png
  - gt_masks_split/<sample>/mask_reference_cube_<frame>.png
  - gt_mapping_<sample>_<frame>.json   （每个 shape 的推断角色 + 置信度 + 触发规则）
  - gt_pending_review.csv              （置信度低 / 规则冲突的帧，绝不静默猜测）

同时为实验一脚本与单元测试导出 convert_labelme_scoped()，返回按 target_scope 选出的二值掩膜。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# ── 路径（相对于本脚本推导，无硬编码样本/文件名） ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
S20_DIR = SCRIPT_DIR.parent  # S20-RAP-FSAM3掩膜生成与验证
DATA_MGMT = S20_DIR.parent  # 07-运行脚本与超参
RECON = DATA_MGMT.parent  # 数据管理
ROOT = RECON.parent.parent  # 00-论文优化重构 → F2DMAS（向上两层）

GT_ROOT = ROOT / "03-GT"  # 含 <sample>/<frame>.json 与 <sample>/<frame>.jpg（原图）

# ── 拆分常量（几何 + 颜色启发式，均可解释） ─────────────────────────────
SMALL_AREA_RATIO = 0.08  # 面积占比 < 8% 视为"小目标"（盆/块）
POT_MIN_CENTROID_Y = 0.55  # 质心低于 55% 图像高度视为"下方位"
POT_GAP_RATIO = 0.04  # 小目标上沿需低于植株主体下沿至少 4% 图高（间隙）
PLANT_UPPER_Y = 0.40  # 植株应延展到图像上 40% 区域
BLUE_MARGIN = 15  # B 通道超出 R/G 至少 15 视为蓝色标定块
PENDING_CONF = 0.5  # 置信度低于此值列入待复核


# ────────────────────────────────────────────────────────────────────────
# 基础工具
# ────────────────────────────────────────────────────────────────────────
def shape_to_mask(shape: dict, height: int, width: int) -> np.ndarray:
    pts = shape.get("points", [])
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(pts, list) or len(pts) < 3:
        return mask
    arr = np.asarray(pts, dtype=np.float32)
    arr = np.rint(arr).astype(np.int32)
    arr[:, 0] = np.clip(arr[:, 0], 0, width - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [arr], 255)
    return mask


class ShapeGeom:
    def __init__(self, index, label, shape_type, n_points, mask, area_px, area_ratio,
                 centroid_y, centroid_y_ratio, min_y, max_y, min_x, max_x, span_x, span_y,
                 aspect, span_ratio, color_bgr=None, role="ignore", confidence=0.0, rule=""):
        self.index = index
        self.label = label
        self.shape_type = shape_type
        self.n_points = n_points
        self.mask = mask
        self.area_px = area_px
        self.area_ratio = area_ratio
        self.centroid_y = centroid_y
        self.centroid_y_ratio = centroid_y_ratio
        self.min_y = min_y
        self.max_y = max_y
        self.min_x = min_x
        self.max_x = max_x
        self.span_x = span_x
        self.span_y = span_y
        self.aspect = aspect
        self.span_ratio = span_ratio
        self.color_bgr = color_bgr
        self.role = role
        self.confidence = confidence
        self.rule = rule

    def to_dict(self) -> dict:
        return {
            "shape_index": self.index + 1,
            "label": self.label,
            "shape_type": self.shape_type,
            "points": self.n_points,
            "area_px": self.area_px,
            "area_ratio": round(self.area_ratio, 5),
            "centroid_y_ratio": round(self.centroid_y_ratio, 4),
            "bbox_x0": self.min_x, "bbox_y0": self.min_y,
            "bbox_x1": self.max_x, "bbox_y1": self.max_y,
            "inferred_role": self.role,
            "confidence": round(self.confidence, 3),
            "rule": self.rule,
            "color_bgr": [round(c, 1) for c in self.color_bgr] if self.color_bgr else None,
        }


def _sample_color(image: np.ndarray | None, mask: np.ndarray) -> tuple[float, float, float] | None:
    if image is None or not mask.any():
        return None
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    patch = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return tuple(float(v) for v in patch.reshape(-1, 3).mean(0))


def _is_blue(color: tuple[float, float, float] | None) -> bool:
    """严格蓝色判定：蓝通道为三通道最大，且相对红/绿明显偏高、红绿之和不反超蓝。

    红褐色花盆（BGR≈[119,85,79]）不满足 (r+g) < b*1.2，会被排除。
    """
    if color is None:
        return False
    b, g, r = color[0], color[1], color[2]
    if not (b >= r and b >= g):
        return False
    if (b - r) < BLUE_MARGIN or (b - g) < BLUE_MARGIN:
        return False
    if (r + g) >= b * 1.2:
        return False
    return True


# ────────────────────────────────────────────────────────────────────────
# 单帧几何 + 颜色推断
# ────────────────────────────────────────────────────────────────────────
def audit_frame(json_path: Path, image: np.ndarray | None = None) -> tuple[list[ShapeGeom], dict]:
    """推断一个 GT json 里每个 shape 的角色（plant/pot/reference_cube/ignore）。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])

    geoms: list[ShapeGeom] = []
    for i, s in enumerate(data.get("shapes", [])):
        m = shape_to_mask(s, h, w)
        area = int((m > 0).sum())
        if area == 0:
            geoms.append(ShapeGeom(i, str(s.get("label", "")), str(s.get("shape_type", "")),
                                   len(s.get("points", [])), m, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0.0, 0.0))
            continue
        ys, xs = np.where(m > 0)
        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        span_x = max_x - min_x + 1
        span_y = max_y - min_y + 1
        cy = float(ys.mean())
        color = _sample_color(image, m)
        g = ShapeGeom(
            index=i, label=str(s.get("label", "")), shape_type=str(s.get("shape_type", "")),
            n_points=len(s.get("points", [])), mask=m, area_px=area,
            area_ratio=area / float(m.size), centroid_y=cy, centroid_y_ratio=cy / h,
            min_y=min_y, max_y=max_y, min_x=min_x, max_x=max_x,
            span_x=span_x, span_y=span_y,
            aspect=(span_y / span_x) if span_x > 0 else 0.0,
            span_ratio=span_x / w, color_bgr=color,
        )
        geoms.append(g)

    # 找植株主体（面积最大者）作为基准；小 shape 不抬高基准
    plant_refs = [g for g in geoms if g.area_px > 0]
    if plant_refs:
        plant_body = max(plant_refs, key=lambda g: g.area_px)
        plant_max_y = plant_body.max_y
        plant_min_y = plant_body.min_y
    else:
        plant_max_y, plant_min_y = h, 0

    for g in geoms:
        if g.area_px == 0:
            g.role = "ignore"
            g.confidence = 1.0
            g.rule = "empty_shape"
            continue
        is_small = g.area_ratio < SMALL_AREA_RATIO
        # 盆/块必须位于植株主体下方且留有清晰间隙（茎与盆体之间）
        below_plant = g.min_y > (plant_max_y + POT_GAP_RATIO * h)
        blue = _is_blue(g.color_bgr)

        if is_small and below_plant:
            if blue:
                g.role = "reference_cube"
                g.confidence = 0.9
                g.rule = "small+below_plant+blue→cube"
            else:
                g.role = "pot"
                conf = 0.9
                if g.aspect < 0.4 or g.aspect > 2.5:
                    conf -= 0.15  # 形状异常（过扁/过高），降置信
                g.confidence = max(0.0, conf)
                g.rule = "small+below_plant+warm→pot"
        else:
            g.role = "plant"
            conf = 0.9
            if g.min_y > PLANT_UPPER_Y * h and not below_plant:
                conf -= 0.3  # 植株未延展到上半区，可能误判
                g.rule = "large_but_not_upper→plant(low_conf)"
            else:
                g.rule = "large+upper→plant"
            g.confidence = max(0.0, conf)

    # 歧义保护：>2 个非空 shape 或多个小目标都 below_plant → 全部降置信并 pending
    non_empty = [g for g in geoms if g.area_px > 0]
    if len(non_empty) > 2:
        for g in geoms:
            g.confidence = min(g.confidence, 0.4)
            g.rule = (g.rule + ";multi_shape_ambiguous") if g.rule else "multi_shape_ambiguous"
    small_lows = [g for g in geoms
                  if g.area_ratio < SMALL_AREA_RATIO and g.min_y > plant_max_y + POT_GAP_RATIO * h]
    if len(small_lows) > 1:
        for g in geoms:
            g.confidence = min(g.confidence, 0.4)
            g.rule = (g.rule + ";multiple_pot_candidates") if g.rule else "multiple_pot_candidates"

    return geoms, {"height": h, "width": w, "shape_count": len(geoms), "plant_max_y": plant_max_y}


def compose_scoped_masks(geoms: list[ShapeGeom], target_scope: str) -> np.ndarray:
    """按口径合成二值掩膜。cube 永不进入 plant / potted GT。"""
    h = geoms[0].mask.shape[0] if geoms else 0
    w = geoms[0].mask.shape[1] if geoms else 0
    plant = np.zeros((h, w), np.uint8)
    pot = np.zeros((h, w), np.uint8)
    cube = np.zeros((h, w), np.uint8)
    for g in geoms:
        if g.role == "plant":
            plant = np.maximum(plant, g.mask)
        elif g.role == "pot":
            pot = np.maximum(pot, g.mask)
        elif g.role == "reference_cube":
            cube = np.maximum(cube, g.mask)
    if target_scope == "plant_only":
        return plant > 0
    if target_scope == "plant_plus_pot":
        return np.logical_or(plant, pot)
    if target_scope == "reference_cube":
        return cube > 0
    return np.logical_or(plant, pot)  # 默认含盆口径


# ────────────────────────────────────────────────────────────────────────
# 对外接口（实验一脚本 / 测试复用）
# ────────────────────────────────────────────────────────────────────────
def convert_labelme_scoped(json_path: Path, target_scope: str = "plant_plus_pot",
                           image_path: Path | None = None) -> tuple[np.ndarray, dict, list[dict]]:
    """等价于实验一 convert_labelme，但返回按口径选出的掩膜（而非无条件 OR）。

    返回 (mask_bool, meta, shape_rows)。
    """
    image = None
    if image_path is not None and image_path.exists():
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is not None:
            image = bgr
    geoms, meta = audit_frame(json_path, image)
    mask = compose_scoped_masks(geoms, target_scope)
    shape_rows = [g.to_dict() for g in geoms]
    meta_out = {
        "width": meta["width"], "height": meta["height"],
        "image_path": str(image_path) if image_path else "",
        "shape_count": meta["shape_count"],
        "gt_area_ratio": float(mask.sum()) / mask.size if mask.size else 0.0,
        "target_scope": target_scope,
    }
    return mask, meta_out, shape_rows


# ────────────────────────────────────────────────────────────────────────
# 批量审计主流程
# ────────────────────────────────────────────────────────────────────────
def iter_gt_frames() -> Iterable[tuple[str, str, Path, Path]]:
    """产出 (sample, frame, json_path, image_path)。"""
    if not GT_ROOT.exists():
        return
    for sample_dir in sorted(GT_ROOT.iterdir()):
        if not sample_dir.is_dir():
            continue
        for json_path in sorted(sample_dir.glob("*.json")):
            frame = json_path.stem
            image_path = sample_dir / f"{frame}.jpg"
            yield sample_dir.name, frame, json_path, image_path


def run_audit(out_root: Path | None = None) -> dict:
    if out_root is None:
        out_root = S20_DIR / "GT口径拆分审计"
    split_dir = out_root / "gt_masks_split"
    split_dir.mkdir(parents=True, exist_ok=True)

    pending_rows: list[dict] = []
    mapping_rows: list[dict] = []
    summary_rows: list[dict] = []

    for sample, frame, json_path, image_path in iter_gt_frames():
        image = None
        if image_path.exists():
            b = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if b is not None:
                image = b
        geoms, meta = audit_frame(json_path, image)
        sdir = split_dir / sample
        sdir.mkdir(parents=True, exist_ok=True)
        for scope, name in [("plant_only", "plant_only"),
                            ("plant_plus_pot", "plant_plus_pot"),
                            ("reference_cube", "reference_cube")]:
            mm = compose_scoped_masks(geoms, scope)
            cv2.imwrite(str(sdir / f"mask_{name}_{frame}.png"), (mm.astype(np.uint8) * 255))

        mapping = {
            "sample": sample, "frame": frame, "target_scope_default": "plant_plus_pot",
            "shapes": [g.to_dict() for g in geoms],
        }
        (sdir / f"gt_mapping_{sample}_{frame}.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        min_conf = min((g.confidence for g in geoms), default=1.0)
        any_cube = any(g.role == "reference_cube" for g in geoms)
        any_pot = any(g.role == "pot" for g in geoms)
        summary_rows.append({
            "sample": sample, "frame": frame, "shape_count": meta["shape_count"],
            "min_confidence": round(min_conf, 3),
            "has_pot": int(any_pot), "has_cube": int(any_cube),
            "pending": int(min_conf < PENDING_CONF),
        })
        if min_conf < PENDING_CONF:
            pending_rows.append({
                "sample": sample, "frame": frame, "min_confidence": round(min_conf, 3),
                "reasons": ";".join(g.rule for g in geoms if g.confidence < PENDING_CONF),
                "note": "需人工核对：启发式置信度低，请勿静默采用",
            })
        for g in geoms:
            mapping_rows.append({
                "sample": sample, "frame": frame, "shape_index": g.index + 1,
                "inferred_role": g.role, "confidence": round(g.confidence, 3), "rule": g.rule,
            })

    _write_csv(out_root / "gt_pending_review.csv",
               ["sample", "frame", "min_confidence", "reasons", "note"], pending_rows)
    _write_csv(out_root / "gt_mapping_summary.csv",
               ["sample", "frame", "shape_index", "inferred_role", "confidence", "rule"], mapping_rows)
    _write_csv(out_root / "gt_audit_summary.csv",
               ["sample", "frame", "shape_count", "min_confidence", "has_pot", "has_cube", "pending"],
               summary_rows)

    print(f"[审计完成] 帧数={len(summary_rows)} 待复核={len(pending_rows)} 输出={out_root}")
    if pending_rows:
        print("\n⚠ 待人工复核帧（未被静默采用）：")
        for r in pending_rows:
            print(f"   {r['sample']}_{r['frame']}  置信度={r['min_confidence']}  原因={r['reasons']}")
    return {"out_root": str(out_root), "frames": len(summary_rows), "pending": len(pending_rows)}


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_audit()
