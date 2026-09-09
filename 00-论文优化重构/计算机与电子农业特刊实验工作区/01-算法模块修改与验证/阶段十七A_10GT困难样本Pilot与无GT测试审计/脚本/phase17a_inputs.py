#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17A — Step 1: Input materialization (deterministic, off the frozen manifests).

Track A (00_input_trackA/, 30 frames):
  For BaiZhang + DouBanLv2, union of the frozen Phase16 context windows
  (context_start..context_end = frame±5) over the 10 GT frames:
    BaiZhang  GT 0033-0037 → union [0028, 0042] = 15 frames
    DouBanLv2 GT 0037-0041 → union [0032, 0046] = 15 frames
  Rationale: A6 consensus and A7 memory must receive REAL temporal context,
  not only the 5 GT frames per sample.

Track B (01_input_trackB/, 32 frames):
  Exactly the 32 rows of the locked Phase16_TEST_manifest.csv
  (no GT is scored here — Track B is a behavioral audit only).

Naming: flat directory with '<sample>_<frame>.jpg' — required by the pipeline's
per-sample grouping key s.rsplit('_',1)[0].

Frames are hardlinked from the raw frame pool (copy fallback on same-dir / cross-fs).
Same files → same materialization (deterministic).
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
PHASE17A = Path(__file__).resolve().parent.parent

P16_SELECT = PHASE16 / "挑战集列表" / "Phase16_selection_manifest_preGT.csv"
P16_TEST = PHASE16 / "挑战集列表" / "Phase16_TEST_manifest.csv"
RAW_FRAMES = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/01-raw_frames")

TRACK_A_IN = PHASE17A / "00_input_trackA"
TRACK_B_IN = PHASE17A / "01_input_trackB"

# 10 PILOT_GT10 frames (frozen DEV material)
GT10 = {
    "BaiZhang": ["0033", "0034", "0035", "0036", "0037"],
    "DouBanLv2": ["0037", "0038", "0039", "0040", "0041"],
}


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def link_frame(sample: str, frame: str, dst_dir: Path) -> Path:
    """Hardlink raw frame → dst_dir/<sample>_<frame>.jpg (copy fallback)."""
    src = RAW_FRAMES / sample / f"{frame}.jpg"
    dst = dst_dir / f"{sample}_{frame}.jpg"
    if dst.exists():
        dst.unlink()
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
    return dst


def build_track_a(rows16: list[dict], dry_run: bool) -> None:
    """Union of context windows per sample → 00_input_trackA/."""
    print("=== Track A: 00_input_trackA/ ===")
    records = []
    for sample in sorted(GT10):
        ctx_min, ctx_max = 10 ** 6, -1
        gt_frames = GT10[sample]
        # context windows come from the frozen Phase16 manifest rows
        for r in rows16:
            if r["sample"] == sample and r["frame"] in gt_frames:
                lo, hi = int(r["context_start"]), int(r["context_end"])
                ctx_min, ctx_max = min(ctx_min, lo), max(ctx_max, hi)
        assert ctx_max >= ctx_min, f"{sample}: no context rows found"
        for f in range(ctx_min, ctx_max + 1):
            frame = f"{f:04d}"
            dt = TRACK_A_IN / f"{sample}_{frame}.jpg"
            src = RAW_FRAMES / sample / f"{frame}.jpg"
            assert src.exists(), f"raw frame missing: {src}"
            records.append({"sample": sample, "frame": frame, "src": str(src), "dst": str(dt)})
            if not dry_run:
                link_frame(sample, frame, TRACK_A_IN)
        print(f"  {sample}: GT {gt_frames[0]}-{gt_frames[-1]} → context [{ctx_min:04d},{ctx_max:04d}] = {ctx_max - ctx_min + 1} frames")
    if not dry_run:
        n = len(list(TRACK_A_IN.glob("*.jpg")))
        print(f"  → linked {n} frames")
    # report
    PHASE17A.joinpath("logs/trackA_input_plan.txt").parent.mkdir(parents=True, exist_ok=True)
    PHASE17A.joinpath("logs/trackA_input_plan.tsv").write_text(
        "\n".join(["\t".join([r["sample"], r["frame"], r["src"], r["dst"]]) for r in records]) + "\n",
        encoding="utf-8")


def build_track_b(rows_test: list[dict], dry_run: bool) -> None:
    """Exactly the 32 locked TEST manifest rows → 01_input_trackB/."""
    print("=== Track B: 01_input_trackB/ ===")
    records = []
    for r in rows_test:
        sample, frame = r["sample"], r["frame"]
        src = Path(r["image_path"])
        dst = TRACK_B_IN / f"{sample}_{frame}.jpg"
        assert src.exists(), f"raw frame missing: {src}"
        records.append({"sample": sample, "frame": frame, "src": str(src), "dst": str(dst)})
        if not dry_run:
            link_frame(sample, frame, TRACK_B_IN)
    if not dry_run:
        n = len(list(TRACK_B_IN.glob("*.jpg")))
        print(f"  → linked {n} frames")
    PHASE17A.joinpath("logs/trackB_input_plan.tsv").write_text(
        "\n".join(["\t".join([r["sample"], r["frame"], r["src"], r["dst"]]) for r in records]) + "\n",
        encoding="utf-8")


def verify_counts() -> None:
    a = len(list(TRACK_A_IN.glob("*.jpg")))
    b = len(list(TRACK_B_IN.glob("*.jpg")))
    print(f"\nVerify: trackA={a} (expect 30)  trackB={b} (expect 32)")
    assert a == 30, f"trackA count {a} != 30"
    assert b == 32, f"trackB count {b} != 32"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only plan, do not link")
    args = ap.parse_args()

    rows16 = read_manifest(P16_SELECT)
    rows_test = read_manifest(P16_TEST)
    assert len(rows_test) == 32, "Phase16 TEST manifest must have 32 rows"

    build_track_a(rows16, args.dry_run)
    build_track_b(rows_test, args.dry_run)
    if not args.dry_run:
        verify_counts()


if __name__ == "__main__":
    main()