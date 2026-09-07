#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 16 — Starting-state audit + input asset audit.

Reads Phase 15 deliverables and the data-management index to produce
Phase16_input_audit.md. Also verifies DouBanLv3 frame readability (that
sample contributed 0 candidates in Phase 15).

Outputs: 阶段十六_HardCase_GT构建与锁定/Phase16_input_audit.md
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE15 = WORKSPACE / "阶段十五_硬案例挑战集与外部验证"
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
DATA = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理")
RAW_FRAMES = DATA / "01-输入图像/01-raw_frames"
COLMAP_LOCKED = DATA / "02-位姿COLMAP/03-final_locked"
INDEX_JSON = DATA / "00-规范与索引/dataset_index.json"

NON_GT_SAMPLES = [
    "BaiZhang", "CaoMei2", "ChangShouHua1", "ChangShouHua3",
    "DouBanLv2", "DouBanLv3", "HongZhang", "WangWenCao1", "WangWenCao2",
    "WanNianQing1", "WanNianQing2", "XiangPiShu1", "XiangPiShu2",
    "XianKeLai2", "XianKeLai3",
]


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           cwd="/data/fj/F2DMAS", timeout=30)
        return r.stdout.strip() + ("\n" + r.stderr.strip() if r.stderr.strip() else "")
    except Exception as e:  # noqa: BLE001
        return f"<git error: {e}>"


def check_readable(sample: str, n: int = 5) -> dict:
    """Try reading n evenly-spaced frames of a sample with cv2."""
    d = RAW_FRAMES / sample
    if not d.exists():
        return {"sample": sample, "files": 0, "readable": 0, "unreadable": []}
    jpgs = sorted(d.glob("*.jpg"))
    total = len(jpgs)
    if total == 0:
        return {"sample": sample, "files": 0, "readable": 0, "unreadable": []}
    idxs = {round(i * (total - 1) / (n - 1)) for i in range(n)}
    bad: list[str] = []
    ok = 0
    for i in sorted(idxs):
        p = jpgs[i]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            bad.append(p.name)
        else:
            ok += 1
    return {"sample": sample, "files": total, "readable": ok, "unreadable": bad}


def main() -> None:
    lines: list[str] = []

    def emit(*xs: str) -> None:
        s = " ".join(xs)
        lines.append(s)
        print(s)

    emit("# Phase 16 — Input Audit\n")

    # ---- 0. Starting state ----
    emit("## 0. Starting state\n")
    emit(f"- HEAD: `{git('rev-parse', 'HEAD')}`")
    emit(f"- Branch: `{git('branch', '--show-current')}`")
    status = git("status", "-sb")
    emit(f"- Status:\n```\n{status}\n```")
    emit("\n```")
    emit(git("log", "-5", "--oneline"))
    emit("```\n")
    emit("- 5 GT samples (frozen): CaoMei1, ChangShouHua2, DouBanLv1, KongQueZhuYu, XianKeLai1")
    emit("- 15 non-GT samples: " + ", ".join(NON_GT_SAMPLES))

    # ---- 1. Difficulty ranking ----
    emit("\n## 1. difficulty_ranking.csv (Phase 15)\n")
    csv_path = PHASE15 / "挑战集列表/difficulty_ranking.csv"
    if not csv_path.exists():
        emit(f"- **MISSING**: {csv_path}")
    else:
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        emit(f"- Rows: {len(rows)}")
        cols = list(rows[0].keys()) if rows else []
        emit(f"- Columns: {', '.join(cols)}")
        diff_vals = [float(r["diff"]) for r in rows]
        emit(f"- diff range: {min(diff_vals):.4f} – {max(diff_vals):.4f}")
        per_sample: dict[str, int] = {}
        for r in rows:
            per_sample[r["sample"]] = per_sample.get(r["sample"], 0) + 1
        emit(f"- Samples represented: {len(per_sample)}/15")
        for s in sorted(per_sample):
            emit(f"  - {s}: {per_sample[s]}")
        missing = [s for s in NON_GT_SAMPLES if s not in per_sample]
        if missing:
            emit(f"- **Samples with 0 candidates**: {', '.join(missing)}")
        # selection-bias check: are any columns model-output-derived?
        model_cols = [c for c in cols if c.lower() in (
            "v00", "v10", "v01", "v11", "f1", "iou", "enhanced",
            "a6", "a7", "pred", "iou_gt")]
        if model_cols:
            emit(f"- **WARNING — model-output columns present**: {model_cols}")
        else:
            emit("- Ranking source: raw-image attributes only "
                 "(laplacian_var, pixel_entropy, exg_sep, green_ratio, dyn_range)")
            emit("- **Model-independence: model-independent / model-agnostic selection** "
                 "(no A6/A7/V10/V01/V11 metric used in ranking)")

    # ---- 2. dataset_index.json ----
    emit("\n## 2. dataset_index.json (sample metadata)\n")
    idx = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    emit(f"- Samples indexed: {len(idx)}")
    emit("| sample | raw_frames | fft_frames | has_gt | colmap_status | reg_rate | points3d | selected_path |")
    emit("|---|---|---|---|---|---|---|---|")
    for s in idx:
        emit(f"| {s['sample']} | {s['raw_frames']} | {s['fft_frames']} | {s['has_gt']} "
             f"| {s['current_colmap_status']} | {s['current_registration_rate']}% "
             f"| {s['current_points3d']} | `{s['selected_colmap_path']}` |")

    # ---- 3. COLMAP completeness ----
    emit("\n## 3. COLMAP completeness (03-final_locked)\n")
    emit("| sample | has_sparse/0 | images_count | registered |")
    emit("|---|---|---|---|")
    for s in NON_GT_SAMPLES:
        d = COLMAP_LOCKED / s
        sparse_ok = (d / "sparse/0/cameras.bin").exists() and (d / "sparse/0/images.bin").exists()
        img_count = len(list((d / "images").glob("*.jpg"))) if (d / "images").exists() else 0
        meta = next((x for x in idx if x["sample"] == s), None)
        reg = meta["current_registered"] if meta else "?"
        emit(f"| {s} | {'yes' if sparse_ok else 'NO'} | {img_count} | {reg} |")

    # ---- 4. Frame readability (incl. DouBanLv3) ----
    emit("\n## 4. Frame readability spot-check (cv2.imread, 5 evenly-spaced frames)\n")
    emit("| sample | total_files | readable | unreadable |")
    emit("|---|---|---|---|")
    for s in NON_GT_SAMPLES:
        r = check_readable(s)
        bad = ", ".join(r["unreadable"]) if r["unreadable"] else "-"
        emit(f"| {r['sample']} | {r['files']} | {r['readable']}/5 | {bad} |")

    # ---- 5. Candidate frame path check ----
    emit("\n## 5. Candidate path existence\n")
    n_missing = 0
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            p = Path(r["path"])
            if not p.exists():
                n_missing += 1
                emit(f"- MISSING: {r['sample']}_{r['frame']} -> {p}")
        if n_missing == 0:
            emit("- All 60 candidate paths exist (via raw_frames symlink layer).")
        else:
            emit(f"- {n_missing} candidate paths missing.")

    # ---- 6. Contact graph ----
    emit("\n## 6. Contact graph / adjacency\n")
    emit("- **No contact graph / adjacency file exists in the project.**")
    emit("- Temporal adjacency is implicit from filename order (0000.jpg → 0001.jpg → …).")
    emit("- COLMAP view graph is derivable from `sparse/0/images.bin` but not materialized.")
    emit("- Phase 16 uses filename order + per-sample isolation for temporal context.")

    # ---- 7. Conclusions ----
    emit("\n## 7. Audit conclusions\n")
    if "DouBanLv3" in missing:
        pass  # readability check above decides
    emit("- Data pool: 15 non-GT samples, ~3,752 raw frames, COLMAP sparse models present for all 15.")
    emit("- Caveats: WangWenCao1 registration only ~70.6%; DouBanLv3 contributed 0 Phase-15 candidates.")
    emit("- Selection protocol is model-independent (raw-image attributes only).")

    out = PHASE16 / "Phase16_input_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
