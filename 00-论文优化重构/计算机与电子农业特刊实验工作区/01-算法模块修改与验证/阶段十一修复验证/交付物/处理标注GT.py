#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从标注好的 GT（/data/fj/F2DMAS/03-GT-区分）生成四层 scope 掩膜。

处理逻辑：
  1. 读取每个 JSON 的 shape，按 label 分组：
     - plant           → 植株掩膜（P2 GT）
     - pot             → 花盆掩膜
     - blue_cube       → 蓝色标定块掩膜
     - potted_plant    → 整体盆栽掩膜（P6 GT）
  2. 若未单独标 plant，则用 potted_plant − pot 自动推算。
  3. 输出四层二值 PNG（每帧4张）到 <out>/gt_masks_split/<sample>/
"""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import csv

GT_SRC = Path("/data/fj/F2DMAS/03-GT-区分")
OUT    = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
              "/S20-RAP-FSAM3掩膜生成与验证/GT口径拆分审计/gt_masks_split_v2")
SCOPES = ["plant", "pot", "potted_plant", "blue_cube"]


def shape_to_mask(shape: dict, h: int, w: int) -> np.ndarray:
    """LabelMe shape → binary mask（支持 polygon/linestrip，闭合后填充）。"""
    pts = shape.get("points", [])
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) < 3:
        return mask
    arr = np.array(pts, dtype=np.float32)
    # linestrip：首尾自动闭合
    if shape.get("shape_type") == "linestrip":
        if abs(arr[0, 0] - arr[-1, 0]) > 5 or abs(arr[0, 1] - arr[-1, 1]) > 5:
            arr = np.vstack([arr, arr[0:1]])
    arr = np.rint(arr).astype(np.int32)
    arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [arr], 255)
    return mask


def process_frame(json_path: Path, image_path: Path | None = None):
    """读取标注 JSON，返回 {scope: mask_bool} + 元信息。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    shapes = data.get("shapes", [])

    masks = {s: np.zeros((h, w), dtype=bool) for s in SCOPES}
    info = {s: 0 for s in SCOPES}

    for sh in shapes:
        lbl = sh.get("label", "").strip()
        m = shape_to_mask(sh, h, w).astype(bool)
        if lbl in masks:
            masks[lbl] |= m
            info[lbl] += 1
        else:
            print(f"  ⚠ 未知 label='{lbl}' in {json_path.name}，跳过")

    # plant 缺失时用 potted_plant − pot 自动推算
    if not masks["plant"].any() and masks["potted_plant"].any():
        masks["plant"] = masks["potted_plant"] & ~masks["pot"]
        info["plant"] = -1  # 标记为自动推算

    return masks, info, (h, w)


def main():
    out_split = OUT
    out_split.mkdir(parents=True, exist_ok=True)

    rows = []
    total_frames = 0
    total_shapes = 0

    for sample_dir in sorted(GT_SRC.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name.startswith("three"):
            continue
        sample = sample_dir.name
        sdir = out_split / sample
        sdir.mkdir(parents=True, exist_ok=True)

        for json_path in sorted(sample_dir.glob("*.json")):
            frame = json_path.stem
            img_path = sample_dir / f"{frame}.jpg"
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR) if img_path.exists() else None

            masks, info, (h, w) = process_frame(json_path, img_path)
            total_frames += 1

            for scope in SCOPES:
                mask_u = (masks[scope].astype(np.uint8) * 255)
                cv2.imwrite(str(sdir / f"mask_{scope}_{frame}.png"), mask_u)
                total_shapes += max(info[scope], 0)

            # 合规性检查：potted_plant 应 = plant ∪ pot
            if masks["potted_plant"].any():
                expected = masks["plant"] | masks["pot"]
                if not np.array_equal(masks["potted_plant"], expected):
                    diff = int((masks["potted_plant"] ^ expected).sum())
                    print(f"  ⚠ {sample}/{frame}: potted_plant ≠ plant|pot (差异像素={diff})")

            rows.append({
                "sample": sample,
                "frame": frame,
                "has_plant": int(masks["plant"].any()),
                "has_pot": int(masks["pot"].any()),
                "has_potted": int(masks["potted_plant"].any()),
                "has_cube": int(masks["blue_cube"].any()),
                "plant_auto": int(info["plant"] == -1),
                "plant_area": int(masks["plant"].sum()),
                "pot_area": int(masks["pot"].sum()),
                "potted_area": int(masks["potted_plant"].sum()),
                "cube_area": int(masks["blue_cube"].sum()),
            })

    # 写汇总 CSV
    fields = ["sample", "frame", "has_plant", "has_pot", "has_potted", "has_cube",
              "plant_auto", "plant_area", "pot_area", "potted_area", "cube_area"]
    csv_path = out_split / "gt_split_summary_v2.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n完成：{total_frames} 帧，{total_shapes} 个形状，输出→ {out_split}")
    print(f"汇总 CSV → {csv_path}")
    print(f"四层 scope：{', '.join(SCOPES)}")
    # plant 自动推算统计
    auto_plant = sum(1 for r in rows if r["plant_auto"])
    print(f"  plant 自动推算（potted−pot）: {auto_plant}/{len(rows)} 帧")


if __name__ == "__main__":
    main()
