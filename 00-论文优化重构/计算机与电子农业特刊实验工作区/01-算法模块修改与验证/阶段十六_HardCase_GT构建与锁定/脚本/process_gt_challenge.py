#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 16 — Challenge GT processing with hard-case QA gates (upgraded from Phase 15).

Reads user labelme annotations from /data/fj/F2DMAS/03-GT-区分_challenge/
and produces:
  GT_potted_clean_challenge/<sample>/mask_potted_clean_<frame>.png
  GT_QA_challenge/gt_audit_challenge.csv
  GT_QA_challenge/challenge_QA_visual/<sample>/<frame>_qa.png

Same target definition as the Phase 12 easy set: potted_clean = potted & ~cube.

Phase 16 additions (hard-case QA gates):
  Q1. Manifest alignment   — only frames listed in the frozen selection manifest
                             (Phase16_selection_manifest_preGT.csv) are accepted;
                             anything else is flagged and skipped.
  Q2. Binary value         — produced mask contains only {0, 255}.
  Q3. Shape exact match    — mask.shape == (imageHeight, imageWidth) from the JSON
                             and == loaded raw-frame shape when available.
  Q4. Non-empty mask       — potted_clean must have ≥ 1 foreground pixel.
  Q5. Not full-image       — potted_clean must not cover > 60% of image pixels.
  Q6. Duplicate checksum   — SHA256 of produced mask compared across all frames;
                             identical checksums are flagged (possible copy-paste).
  Q7. Naming alignment     — every <frame>.json must pair with <frame>.jpg in the
                             same GT directory.
  Q8. Foreground ratio     — sanity flag column (fg_ratio = clean_area / (h*w)).

Usage:
    python process_gt_challenge.py
      [--gt_src /data/fj/F2DMAS/03-GT-区分_challenge]
      [--manifest 阶段十六/挑战集列表/Phase16_selection_manifest_preGT.csv]
      [--out_root 阶段十六根目录 (default: 本脚本所在阶段十六根目录)]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PHASE16 = Path(__file__).resolve().parent.parent  # 阶段十六根目录
DEFAULT_MANIFEST = PHASE16 / "挑战集列表/Phase16_selection_manifest_preGT.csv"
GT_SRC_DEFAULT = Path("/data/fj/F2DMAS/03-GT-区分_challenge")


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


def load_manifest(manifest_path: Path) -> set[tuple[str, str]]:
    """Return {(sample, frame)} of expected GT targets."""
    if not manifest_path.exists():
        return set()
    out: set[tuple[str, str]] = set()
    with open(manifest_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("is_gt_target") == "yes":
                out.add((r["sample"], r["frame"]))
    return out


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

    potted_clean = potted & ~cube
    overlap = int((potted & cube).sum())

    if not cube.any():
        cube_status = "absent"
    elif overlap > 0:
        cube_status = f"overlapped_{overlap}px"
    else:
        cube_status = "clean"

    pot_status = "present" if pot.any() else "missing"

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
    h, w = shape
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    vis[:, :w] = img

    panel2 = img.copy()
    potted_clean = masks["potted_clean"]
    cube = masks["cube"]
    conflict = masks["potted"] & masks["cube"]

    green = panel2.copy()
    green[potted_clean] = [0, 200, 0]
    panel2 = cv2.addWeighted(panel2, 0.5, green, 0.5, 0)

    if cube.any():
        contours, _ = cv2.findContours(cube.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel2, contours, -1, (255, 100, 0), 3)

    if conflict.any():
        red = panel2.copy()
        red[conflict] = [0, 0, 255]
        panel2 = cv2.addWeighted(panel2, 0.7, red, 0.3, 0)

    vis[:, w:w*2] = panel2

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_src", type=str, default=str(GT_SRC_DEFAULT))
    ap.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out_root", type=str, default=str(PHASE16))
    args = ap.parse_args()

    GT_SRC = Path(args.gt_src)
    OUT_ROOT = Path(args.out_root)
    CLEAN_DIR = OUT_ROOT / "GT_potted_clean_challenge"
    QA_DIR = OUT_ROOT / "GT_QA_challenge"
    VIS_DIR = QA_DIR / "challenge_QA_visual"

    manifest_targets = load_manifest(Path(args.manifest))
    manifest_label = Path(args.manifest).name if Path(args.manifest).exists() else "<missing manifest>"

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    pending = []
    skipped = []          # Q1 / Q7 rejections
    qa_fail = []          # Q4 / Q5 hard failures
    checksums: dict[str, list[str]] = Counter()   # Q6: sha -> [stem,...]

    for sample_dir in sorted(GT_SRC.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name.startswith("three"):
            continue
        sample = sample_dir.name
        s_clean = CLEAN_DIR / sample
        s_clean.mkdir(parents=True, exist_ok=True)
        s_vis = VIS_DIR / sample
        s_vis.mkdir(parents=True, exist_ok=True)

        jpgs = {p.stem for p in sample_dir.glob("*.jpg")}

        for json_path in sorted(sample_dir.glob("*.json")):
            frame = json_path.stem
            stem = f"{sample}_{frame}"

            # Q7: naming alignment — .json must pair with .jpg
            if frame not in jpgs:
                skipped.append({"stem": stem, "qa": "Q7_naming_no_jpg"})
                print(f"  [Q7] {stem}: JSON without matching .jpg — skipped")
                continue

            # Q1: manifest alignment
            if manifest_targets and (sample, frame) not in manifest_targets:
                skipped.append({"stem": stem, "qa": "Q1_not_in_manifest"})
                print(f"  [Q1] {stem}: not in {manifest_label} — skipped")
                continue

            img_path = sample_dir / f"{frame}.jpg"
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            result = process_frame(json_path, img_path)
            meta = result["masks"]
            mask_clean = meta["potted_clean"]
            shape = result["shape"]
            m_h, m_w = shape

            # ---- Phase 16 QA gates ----
            row = {"sample": sample, "frame": frame, **result["meta"]}

            # Q3: shape exact match vs JSON dims (inherent) and vs raw image when loaded
            img_h, img_w = (0, 0)
            if img is not None:
                img_h, img_w = img.shape[:2]
            row["img_shape"] = f"{img_h}x{img_w}" if img is not None else "unloaded"
            if img is not None and (img_h, img_w) != (m_h, m_w):
                row["qa_shape_mismatch"] = 1
                qa_fail.append({"stem": stem, "qa": "Q3_shape_mismatch_img"})
            else:
                row["qa_shape_mismatch"] = 0

            # Q4: non-empty
            if not result["meta"]["formal_p6_valid"]:
                row["qa_empty"] = 1
                qa_fail.append({"stem": stem, "qa": "Q4_empty_mask"})
            else:
                row["qa_empty"] = 0

            # Q5: not full-image (> 60%)
            fg_ratio = result["meta"]["clean_area"] / (m_h * m_w)
            row["fg_ratio"] = f"{fg_ratio:.6f}"
            row["qa_too_big"] = 1 if fg_ratio > 0.60 else 0
            if fg_ratio > 0.60:
                qa_fail.append({"stem": stem, "qa": "Q5_full_image"})

            # ---- persist mask + overlay ----
            clean_u = (meta["potted_clean"].astype(np.uint8) * 255)
            # Q2: binary value validation (should always hold by construction)
            uniq = set(np.unique(clean_u))
            row["binary_ok"] = 1 if uniq <= {0, 255} else 0
            if row["binary_ok"] == 0:
                qa_fail.append({"stem": stem, "qa": "Q2_non_binary"})

            mask_file = s_clean / f"mask_potted_clean_{frame}.png"
            cv2.imwrite(str(mask_file), clean_u)

            # Q6: duplicate checksum
            sha = hashlib.sha256(clean_u.tobytes()).hexdigest()[:16]
            row["mask_sha256_16"] = sha
            checksums[sha].append(stem)

            if img is not None:
                vis = draw_overlay(img, result["masks"], shape)
                cv2.imwrite(str(s_vis / f"{frame}_qa.png"), vis)

            rows.append(row)

            if row["pot_status"] == "missing" or row["cube_status"] == "absent":
                pending.append(row)

    # ---- QA audit CSV ----
    fields = ["sample", "frame", "has_potted", "has_manual_plant", "plant_auto",
              "has_pot", "has_cube", "cube_status", "pot_status",
              "potted_area", "clean_area", "cube_area", "potted_cube_overlap",
              "formal_p6_valid", "formal_p2_valid", "reason",
              "img_shape", "qa_shape_mismatch", "qa_empty", "qa_too_big",
              "fg_ratio", "binary_ok", "mask_sha256_16"]
    qa_csv = QA_DIR / "gt_audit_challenge.csv"
    with qa_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    pending_csv = QA_DIR / "pending_review_challenge.csv"
    with pending_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(pending)

    dup_stems: list[str] = []
    for sha, stems in checksums.items():
        if len(stems) > 1:
            dup_stems.append(f"{sha}:{'|'.join(stems)}")

    # ---- summary ----
    p6_valid = sum(1 for r in rows if r["formal_p6_valid"])
    p2_valid = sum(1 for r in rows if r["formal_p2_valid"])
    cube_overlaps = sum(1 for r in rows if r["potted_cube_overlap"] > 0)
    print(f"完成：{len(rows)} 帧（manifest targets: {len(manifest_targets)}）")
    print(f"  P6 有效：{p6_valid}/{len(rows)}")
    print(f"  P2 有效：{p2_valid}/{len(rows)}")
    print(f"  cube 重叠已消除：{cube_overlaps} 帧")
    print(f"  待复核：{len(pending)} 帧")
    print(f"  QA hard-fail：{len(qa_fail)} 帧  -> {[f['stem']+':'+f['qa'] for f in qa_fail]}")
    print(f"  Skipped：{len(skipped)} 帧  -> {[s['stem']+':'+s['qa'] for s in skipped]}")
    if dup_stems:
        print(f"  ** 重复 checksum（可疑复制）：{len(dup_stems)} 组 -> {dup_stems}")
    else:
        print("  duplicate checksum：无")
    print(f"  输出→ {CLEAN_DIR.name}/, {QA_DIR.name}/")


if __name__ == "__main__":
    main()
