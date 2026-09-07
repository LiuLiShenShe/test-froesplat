#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 16 — Annotation package generation.

For each GT target in Phase16_selection_manifest_preGT.csv, build a self-contained
annotation folder under annotation_package/ with:
  <challenge_id>/target.jpg           — the frame to annotate (copy)
  <challenge_id>/context_prev.jpg     — frame at context_start (temporal context)
  <challenge_id>/context_next.jpg     — frame at context_end   (temporal context)
  <challenge_id>/instruction_card.md  — difficulty tag, identity hint, path
Per-sample contact sheet (grid of all target frames with difficulty labels) is
written to annotation_package/contact_sheets/<sample>_contact.jpg.

NO model outputs (V00/V10/V01/V11) are placed anywhere in the package to avoid
annotation bias.

Usage:
    python phase16_annotation_package.py
        [--manifest .../Phase16_selection_manifest_preGT.csv]
        [--out annotation_package]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PHASE16 = Path(__file__).resolve().parent.parent
RAW_FRAMES = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/01-raw_frames")
DEFAULT_MANIFEST = PHASE16 / "挑战集列表/Phase16_selection_manifest_preGT.csv"

DIFF_LABELS = {
    "D1_neighbor_plant_adhesion": "邻居植株粘连",
    "D3_foreground_background_confusion": "前景/背景混淆",
    "D6_small_target": "目标过小",
    "D11_cluttered_background": "背景杂乱",
    "D12_low_contrast": "低对比度",
    "D14_other": "其他",
}


def nearest_readable(raw_dir: Path, base: int, window: tuple[int, int], total: int) -> Path | None:
    """Return path of the readable frame nearest `base` within [window[0], window[1]].

    Scans outward from `base` within the clamped window; returns None if nothing
    in the window is readable. Used so an unreadable context frame degrades to
    the closest usable one instead of leaving the package without context.
    """
    lo, hi = window
    for d in range(0, max(hi - base, base - lo) + 1):
        for cand in (base - d, base + d):
            if cand < lo or cand > hi or cand < 0 or cand >= total:
                continue
            p = raw_dir / f"{cand:04d}.jpg"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    return p
    return None


def build_instruction_card(challenge_id: str, row: dict, target_abs: Path,
                           prev_abs: Path | None, next_abs: Path | None) -> str:
    prim = DIFF_LABELS.get(row["primary_difficulty"], row["primary_difficulty"])
    sec = DIFF_LABELS.get(row["secondary_difficulty"], row["secondary_difficulty"]) \
        if row["secondary_difficulty"] else "无"
    prev_note = "（可用）" if prev_abs is not None else "（窗口内无可用帧，跳过）"
    next_note = "（可用）" if next_abs is not None else "（窗口内无可用帧，跳过）"
    prev_s = str(prev_abs) if prev_abs is not None else "-"
    next_s = str(next_abs) if next_abs is not None else "-"
    return f"""# Instruction Card — {challenge_id}

## 标注对象
- **sample / 序列**: {row['sample']}
- **目标帧**: {row['frame']}.jpg
- **split**: {row['split']}（DEV / TEST — 仅用于分组，不影响标注内容）

## 难度提示（启发式，标注时请人工确认）
- primary:   {prim} ({row['primary_difficulty']})
- secondary: {sec}
- raw-image difficulty score: {row['difficulty_score']}
- Phase-15 排名: {row['difficulty_rank']}

## 目标身份规则（关键）
1. 本 sample 是时间序列，目标是**同一株个体植物**（跨帧一致）。
2. 先查看 context_prev.jpg / context_next.jpg 确认目标个体身份。
3. 只标注目标个体；相邻植株、背景植被、温室结构一律排除。

## 需要标注的 label
| label | 必需? | 说明 |
|---|---|---|
| `potted_plant` | **必须** | 目标整株（含盆），闭合 linestrip |
| `plant` | 建议 | 仅叶+茎（不含盆） |
| `pot` | 建议 | 花盆/托盘 |
| `blue_cube` | 如存在 | 蓝色标定块（potted_clean 会自动扣除） |

## 原始帧路径（标注完成后，把 .jpg + .json 放同目录）
- 原始帧: `{target_abs}`
- 上下文前帧{prev_note}: `{prev_s}`
- 上下文后帧{next_note}: `{next_s}`
- **GT 输出目录**: `/data/fj/F2DMAS/03-GT-区分_challenge/{row['sample']}/`

## 尺寸
图像 3840×2160（高×宽，纵向；JSON 中 imageHeight=3840, imageWidth=2160），
shape_type=linestrip，`imagePath` 字段只填帧名如 `{row['frame']}.jpg`。

## 若无法确定身份
在 JSON 的 text 字段写明 "ambiguous"，并把该帧从 manifest 移除（通知负责人）。
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", type=str, default=str(PHASE16 / "annotation_package"))
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_root = Path(args.out)
    contact_dir = out_root / "contact_sheets"

    with open(manifest_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    targets = [r for r in rows if r.get("is_gt_target") == "yes"]
    print(f"Targets to package: {len(targets)}")

    missing = []
    # group by sample for contact sheets
    from collections import OrderedDict
    by_sample: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in sorted(targets, key=lambda r: (r["sample"], r["frame"])):
        by_sample.setdefault(r["sample"], []).append(r)

    for sample, srows in by_sample.items():
        raw_dir = RAW_FRAMES / sample
        if not raw_dir.exists():
            print(f"  [warn] missing raw dir: {raw_dir}")
            continue

        # ---- per-target folders ----
        thumbs: list[np.ndarray] = []
        labels: list[str] = []
        for r in srows:
            cid = r["challenge_id"]
            target_abs = raw_dir / f"{r['frame']}.jpg"
            if not target_abs.exists():
                missing.append(str(target_abs))
                print(f"  [warn] missing target frame: {target_abs}")
                continue

            ctx_start = int(r["context_start"])
            ctx_end = int(r["context_end"])
            target_idx = int(r["frame"])
            total = len(list(raw_dir.glob("*.jpg"))) or (ctx_end + 1)

            # context frames: prefer exact window edges, fall back to the nearest
            # readable frame inside the window (a few frames here are corrupt)
            prev_abs = nearest_readable(raw_dir, ctx_start,
                                        (max(0, target_idx - 5), max(target_idx, ctx_start)),
                                        total)
            next_abs = nearest_readable(raw_dir, ctx_end,
                                        (min(target_idx, ctx_end), min(total - 1, target_idx + 5)),
                                        total)

            folder = out_root / cid
            folder.mkdir(parents=True, exist_ok=True)

            img = cv2.imread(str(target_abs), cv2.IMREAD_COLOR)
            if img is None:
                missing.append(str(target_abs))
                print(f"  [warn] unreadable target frame: {target_abs}")
                continue
            cv2.imwrite(str(folder / "target.jpg"), img)

            prev_img = cv2.imread(str(prev_abs), cv2.IMREAD_COLOR) if prev_abs else None
            if prev_img is not None:
                cv2.imwrite(str(folder / "context_prev.jpg"), prev_img)

            next_img = cv2.imread(str(next_abs), cv2.IMREAD_COLOR) if next_abs else None
            if next_img is not None:
                cv2.imwrite(str(folder / "context_next.jpg"), next_img)

            card = build_instruction_card(cid, r, target_abs, prev_abs, next_abs)
            (folder / "instruction_card.md").write_text(card, encoding="utf-8")

            # thumb for contact sheet (downscale)
            h, w = img.shape[:2]
            scale = 320 / w
            tw, th = 320, int(h * scale)
            thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            # difficulty text header strip
            head = np.full((28, tw, 3), 255, dtype=np.uint8)
            prim = r["primary_difficulty"].split("_")[0]
            cv2.putText(head, f"{cid}  {prim} D={r['difficulty_score']}",
                        (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
            thumbs.append(np.vstack([head, thumb]))
            labels.append(cid)

        # ---- contact sheet per sample (up to 4 targets → 2×2 grid) ----
        if not thumbs:
            continue
        n = len(thumbs)
        cols = 2 if n > 2 else n
        rows_n = (n + cols - 1) // cols
        cw = max(t.shape[1] for t in thumbs)
        ch = max(t.shape[0] for t in thumbs)
        sheet = np.full((rows_n * ch, cols * cw, 3), 200, dtype=np.uint8)
        for i, t in enumerate(thumbs):
            r_, c_ = divmod(i, cols)
            sheet[r_ * ch:(r_ + 1) * ch, c_ * cw:(c_ + 1) * cw] = t
        contact_dir.mkdir(parents=True, exist_ok=True)
        cs_path = contact_dir / f"{sample}_contact.jpg"
        cv2.imwrite(str(cs_path), sheet)
        print(f"  [{sample}] packaged {n} targets, contact → {cs_path.name}")

    print(f"\nTotal packaged: {len(targets) - len(missing)}/{len(targets)}")
    if missing:
        print("Missing/unreadable frames:")
        for m in missing:
            print(f"  {m}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
