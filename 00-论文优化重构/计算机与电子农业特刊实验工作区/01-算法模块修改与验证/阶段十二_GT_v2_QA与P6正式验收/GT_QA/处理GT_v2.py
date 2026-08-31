#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段十二 GT v2 QA：读取标注好的GT，生成potted_clean、audit CSV、21帧可视化。

输入：/data/fj/F2DMAS/03-GT-区分/<sample>/<frame>.json
输出：
  GT_potted_clean/<sample>/mask_potted_clean_<frame>.png  (potted & ~cube)
  GT_QA/gt_audit_v2_qa.csv
  GT_QA/21帧可视化/<sample>/<frame>_qa.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

GT_SRC = Path("/data/fj/F2DMAS/03-GT-区分")
DELIVER = Path(__file__).resolve().parent.parent  # 阶段十二根目录
CLEAN_DIR = DELIVER / "GT_potted_clean"
QA_DIR = DELIVER / "GT_QA"
VIS_DIR = QA_DIR / "21帧可视化"
FAIL_FRAMES = {("CaoMei1", "0100"), ("ChangShouHua2", "0100"), ("DouBanLv1", "0000")}


def shape_to_mask(shape: dict, h: int, w: int) -> np.ndarray:
    pts = shape.get("points", [])
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) < 3:
        return mask
    arr = np.array(pts, dtype=np.float32)
    if shape.get("shape_type") == "linestrip":
        if abs(arr[0, 0] - arr[-1, 0]) > 5 or abs(arr[0, 1] - arr[-1, 1]) > 5:
            arr = np.vstack([arr, arr[0:1]])
    arr = np.rint(arr).astype(np.int32)
    arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [arr], 255)
    return mask


def process_frame(json_path: Path, img_path: Path | None):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h, w = int(data["imageHeight"]), int(data["imageWidth"])
    shapes = data.get("shapes", [])

    merged = {}
    for sh in shapes:
        lbl = sh.get("label", "").strip()
        m = shape_to_mask(sh, h, w).astype(bool)
        merged.setdefault(lbl, np.zeros((h, w), dtype=bool))
        merged[lbl] |= m

    potted = merged.get("potted_plant", np.zeros((h, w), dtype=bool))
    cube = merged.get("blue_cube", np.zeros((h, w), dtype=bool))
    plant = merged.get("plant", np.zeros((h, w), dtype=bool))
    pot = merged.get("pot", np.zeros((h, w), dtype=bool))

    has_manual_plant = bool(plant.any())
    if not has_manual_plant and potted.any() and pot.any():
        plant = potted & ~pot  # derived

    # potted_clean = potted & ~cube
    overlap = int((potted & cube).sum())
    potted_clean = potted & ~cube

    # cube_status
    if not cube.any():
        cube_status = "absent"
    elif overlap > 0:
        cube_status = f"overlapped_{overlap}px"
    else:
        cube_status = "clean"

    # pot_status
    pot_status = "present" if pot.any() else "missing"

    # formal validity
    formal_p6 = bool(potted_clean.any())
    formal_p2 = bool(plant.any()) and has_manual_plant and pot.any()

    reason_parts = []
    if not has_manual_plant:
        reason_parts.append("plant_auto_derived")
    if not pot.any():
        reason_parts.append("pot_missing")
    if not cube.any():
        reason_parts.append("cube_absent")
    if overlap > 0:
        reason_parts.append(f"cube_overlap_removed_{overlap}px")

    return {
        "masks": {"potted": potted, "cube": cube, "plant": plant, "pot": pot, "potted_clean": potted_clean},
        "meta": {
            "has_potted": bool(potted.any()),
            "has_manual_plant": has_manual_plant,
            "plant_auto": int(not has_manual_plant),
            "has_pot": bool(pot.any()),
            "has_cube": bool(cube.any()),
            "cube_status": cube_status,
            "pot_status": pot_status,
            "potted_area": int(potted.sum()),
            "clean_area": int(potted_clean.sum()),
            "cube_area": int(cube.sum()),
            "potted_cube_overlap": overlap,
            "formal_p6_valid": formal_p6,
            "formal_p2_valid": formal_p2,
            "reason": ";".join(reason_parts) if reason_parts else "ok",
        },
        "shape": (h, w),
    }


def draw_overlay(img, masks, shape):
    """3-panel visualization: 原图 | potted_clean+cube+冲突 | plant+pot"""
    h, w = shape
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    vis[:, :w] = img

    # Panel 2: potted_clean (green) + cube (blue) + conflict (red)
    panel2 = img.copy()
    potted_clean = masks["potted_clean"]
    cube = masks["cube"]
    conflict = masks["potted"] & masks["cube"]

    # green overlay for potted_clean
    green = panel2.copy()
    green[potted_clean] = [0, 200, 0]
    panel2 = cv2.addWeighted(panel2, 0.5, green, 0.5, 0)

    # blue contour for cube
    if cube.any():
        contours, _ = cv2.findContours(cube.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel2, contours, -1, (255, 100, 0), 3)

    # red for conflict
    if conflict.any():
        red = panel2.copy()
        red[conflict] = [0, 0, 255]
        panel2 = cv2.addWeighted(panel2, 0.7, red, 0.3, 0)

    vis[:, w:w*2] = panel2

    # Panel 3: plant (bright green) + pot (orange)
    panel3 = img.copy()
    plant = masks["plant"]
    pot = masks["pot"]

    if plant.any():
        green = panel3.copy()
        green[plant] = [0, 255, 100]
        panel3 = cv2.addWeighted(panel3, 0.5, green, 0.5, 0)
    if pot.any():
        orange = panel3.copy()
        orange[pot] = [0, 140, 255]
        panel3 = cv2.addWeighted(panel3, 0.5, orange, 0.5, 0)

    vis[:, w*2:] = panel3
    return vis


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    pending = []

    for sample_dir in sorted(GT_SRC.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name.startswith("three"):
            continue
        sample = sample_dir.name
        s_clean = CLEAN_DIR / sample
        s_clean.mkdir(parents=True, exist_ok=True)
        s_vis = VIS_DIR / sample
        s_vis.mkdir(parents=True, exist_ok=True)

        for json_path in sorted(sample_dir.glob("*.json")):
            frame = json_path.stem
            img_path = sample_dir / f"{frame}.jpg"
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR) if img_path.exists() else None

            result = process_frame(json_path, img_path)
            meta = result["meta"]
            masks = result["masks"]
            shape = result["shape"]

            # Save potted_clean mask
            clean_u = (masks["potted_clean"].astype(np.uint8) * 255)
            cv2.imwrite(str(s_clean / f"mask_potted_clean_{frame}.png"), clean_u)

            # Save visualization
            if img is not None:
                vis = draw_overlay(img, masks, shape)
                cv2.imwrite(str(s_vis / f"{frame}_qa.png"), vis)

            # CSV row
            row = {"sample": sample, "frame": frame, **meta}
            rows.append(row)

            if meta["pot_status"] == "missing" or meta["cube_status"] == "absent":
                pending.append(row)

    # Write QA CSV
    fields = ["sample", "frame", "has_potted", "has_manual_plant", "plant_auto",
              "has_pot", "has_cube", "cube_status", "pot_status",
              "potted_area", "clean_area", "cube_area", "potted_cube_overlap",
              "formal_p6_valid", "formal_p2_valid", "reason"]
    qa_csv = QA_DIR / "gt_audit_v2_qa.csv"
    with qa_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Write pending review
    pending_csv = QA_DIR / "pending_review.csv"
    with pending_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(pending)

    # Summary
    p6_valid = sum(1 for r in rows if r["formal_p6_valid"])
    p2_valid = sum(1 for r in rows if r["formal_p2_valid"])
    cube_overlaps = sum(1 for r in rows if r["potted_cube_overlap"] > 0)
    print(f"完成：{len(rows)} 帧")
    print(f"  P6 有效：{p6_valid}/{len(rows)}")
    print(f"  P2 有效：{p2_valid}/{len(rows)}")
    print(f"  cube 重叠已消除：{cube_overlaps} 帧")
    print(f"  待复核：{len(pending)} 帧")
    print(f"  输出→ {DELIVER.name}/GT_potted_clean/, GT_QA/")


if __name__ == "__main__":
    main()
